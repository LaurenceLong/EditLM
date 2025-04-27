# src/train_sft_hf.py
import argparse
import os
import warnings
import math
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader
import random
from termcolor import colored

# 确保导入所有必要的组件
from data import (
    collate_fn,
    DeletionTaskDataset,
    InsertionTaskDataset,
    PredictionTaskDataset
)
from model_hf import EditLMHF
from tokenizer import get_tokenizer, get_del_token_id
from utils import save_ckpt, WarmupCosine, load_model_from_ckpt


def set_trainable_parts(model: EditLMHF, phase: str, freeze_backbone=True):
    """
    根据训练阶段设置模型的可训练部分。

    Args:
        model: EditLMHF模型
        phase: 当前训练阶段 ('editing')
        freeze_backbone: 是否冻结基础模型
    """
    # 首先根据freeze_backbone参数决定是否冻结backbone
    if freeze_backbone:
        # 冻结backbone
        for param in model.backbone.parameters():
            param.requires_grad_(False)
        print(colored("已冻结backbone参数", "yellow"))
    else:
        # 保持backbone可训练
        for param in model.backbone.parameters():
            param.requires_grad_(True)
        print(colored("保持backbone参数可训练", "green"))

    if phase == 'editing':
        print(colored("设置编辑阶段的可训练部分（编辑头/投影层/嵌入）", "cyan"))

        # 确保编辑相关的组件可训练
        trainable_components = [
            model.index_head, model.edit_head,
            model.triple_proj, model.fuse_proj
        ]

        # 处理可学习的embedding参数
        if hasattr(model, 'gap_token_embed') and isinstance(model.gap_token_embed, nn.Parameter):
            model.gap_token_embed.requires_grad_(True)
            print("已解冻 gap_token_embed")
        elif hasattr(model, 'gap_token_embed'):  # 如果是module
            trainable_components.append(model.gap_token_embed)

        if hasattr(model, 'boundary_embed') and isinstance(model.boundary_embed, nn.Parameter):
            model.boundary_embed.requires_grad_(True)
            print("已解冻 boundary_embed")
        elif hasattr(model, 'boundary_embed'):  # 如果是module
            trainable_components.append(model.boundary_embed)

        # 计算可训练参数
        unfrozen_count = 0
        component_names = ["index_head", "edit_head", "triple_proj", "fuse_proj", "gap_token_embed", "boundary_embed"]
        for i, component in enumerate(trainable_components):
            if component is not None:
                if isinstance(component, nn.Module):
                    component_name = component_names[i] if i < len(component_names) else type(component).__name__
                    count_before = unfrozen_count
                    for param in component.parameters():
                        param.requires_grad_(True)
                        unfrozen_count += param.numel()
                    if unfrozen_count > count_before:
                        print(f"已解冻 {component_name} ({unfrozen_count - count_before} 参数)")

        # 添加单独处理的Parameter计数
        if hasattr(model, 'gap_token_embed') and isinstance(model.gap_token_embed,
                                                            nn.Parameter) and model.gap_token_embed.requires_grad:
            unfrozen_count += model.gap_token_embed.numel()
        if hasattr(model, 'boundary_embed') and isinstance(model.boundary_embed,
                                                           nn.Parameter) and model.boundary_embed.requires_grad:
            unfrozen_count += model.boundary_embed.numel()
    else:
        warnings.warn(f"未知的训练阶段 '{phase}'，没有设置可训练参数。")

    # 输出最终的可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(colored(
        f"阶段 '{phase}': {trainable_params / 1e6:.2f}M 可训练参数，共 {total_params / 1e6:.2f}M 总参数。", "cyan"))


def get_next_batch(loader_iter, loader, task_type, step):
    """获取下一个批次数据，处理迭代器耗尽的情况"""
    try:
        return next(loader_iter), loader_iter
    except StopIteration:
        print(f"\n重置 {task_type} 数据加载器迭代器，步骤 {step}")
        new_iter = iter(loader)
        return next(new_iter), new_iter


def main():
    ap = argparse.ArgumentParser(
        description="使用交替的删除、插入和预测任务训练EditLM")
    ap.add_argument("--outdir", required=True, help="模型保存目录")
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B",
                    help="HF checkpoint，例如 Qwen/Qwen2.5-0.5B")
    ap.add_argument("--from_file", help="从检查点文件路径加载（恢复训练状态）")
    ap.add_argument("--data_dir", default="./data/wikitext_processed",
                    help="预处理数据目录（应包含deletion/、insertion/和*_prediction_inputs.pt）")
    ap.add_argument("--batch_size", type=int, default=8, help="训练批次大小（每种任务类型）")
    ap.add_argument("--steps", type=int, default=100000, help="总训练步数（所有任务）")
    ap.add_argument("--freeze", action="store_true", default=False, help="是否冻结backbone参数")
    ap.add_argument("--skip_prediction", action="store_true", default=False, help="是否跳过预测任务")
    ap.add_argument("--prediction_epochs", type=int, default=1, help="预测任务训练的轮数")
    ap.add_argument("--lr", type=float, default=2e-4, help="学习率")
    ap.add_argument("--grad_accum", type=int, default=4, help="梯度累积步数")
    ap.add_argument("--warmup_steps", type=int, default=1000, help="预热步数")
    ap.add_argument("--ckpt_every", type=int, default=5000, help="每多少步保存检查点")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    args = ap.parse_args()

    # 设置随机种子
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 创建输出目录
    os.makedirs(args.outdir, exist_ok=True)

    class Config:
        batch_size = args.batch_size
        total_steps = args.steps
        lr = args.lr
        warmup_steps = args.warmup_steps
        grad_clip = 1.0
        fp16 = True
        ckpt_every = args.ckpt_every
        seq_len = None  # 将从元数据加载
        grad_accum_steps = args.grad_accum
        skip_prediction = args.skip_prediction

    cfg = Config()

    # 加载元数据获取序列长度
    try:
        found_meta = False
        potential_paths = [
            os.path.join(args.data_dir, "train_metadata.pt"),
            os.path.join(args.data_dir, "deletion", "train_metadata.pt"),
            os.path.join(args.data_dir, "insertion", "train_metadata.pt")
        ]
        for path in potential_paths:
            if os.path.exists(path):
                metadata = torch.load(path)
                if "seq_len" in metadata:
                    cfg.seq_len = metadata["seq_len"]
                    print(colored(f"从元数据加载序列长度 {cfg.seq_len}: {path}", "green"))
                    found_meta = True
                    break
                else:
                    print(colored(f"警告：在元数据文件中未找到'seq_len': {path}", "yellow"))

        if not found_meta:
            raise FileNotFoundError("找不到包含'seq_len'的有效元数据文件")

    except FileNotFoundError as e:
        print(colored(f"错误: {e}", "red"))
        print("请确保预处理已完成并且data_dir路径正确。")
        return
    except KeyError:
        print(colored(f"错误：在加载的元数据文件中未找到'seq_len'。", "red"))
        return
    except Exception as e:
        print(colored(f"加载元数据时发生意外错误: {e}", "red"))
        return

    # 设备检测
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.fp16 = cfg.fp16 and (device == "cuda")
    print(colored(f"使用设备: {device}, FP16: {cfg.fp16}", "cyan"))

    # 模型和Tokenizer加载
    print(colored(f"加载模型: {args.base}", "cyan"))
    tokenizer = None
    model = None
    start_step = 0

    # 尝试从检查点加载
    if args.from_file:
        print(colored(f"加载检查点: {args.from_file}", "cyan"))
        try:
            model, tokenizer, start_step = load_model_from_ckpt(args.from_file, base_model=args.base)
            model.to(device)
            print(colored(f"模型从检查点加载完成，将从步骤 {start_step} 继续训练。", "green"))
        except Exception as e:
            print(colored(f"从检查点加载失败: {e}. 将尝试从基础模型重新开始。", "red"))
            args.from_file = None
            start_step = 0

    # 如果未从检查点加载，则创建新模型
    if model is None:
        print("加载新的 Tokenizer 和 Model...")
        tokenizer = get_tokenizer(args.base, use_fast=True)
        model = EditLMHF(base_model=args.base, index_loss_weight=1.0).to(device)

    del_token_id = get_del_token_id(tokenizer)  # 用于DeletionTaskDataset

    # 优化器和调度器
    print(colored("创建优化器和调度器...", "cyan"))
    # 在创建优化器前设置可训练部分，确保只对可训练部分应用梯度更新
    set_trainable_parts(model, 'editing', freeze_backbone=args.freeze)

    # 只对需要更新的参数创建优化器，减少内存占用
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        trainable_params,
        lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    sched = WarmupCosine(opt, cfg, cfg.total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)

    # 加载优化器状态（如果从检查点恢复）
    if args.from_file and start_step > 0:
        try:
            ckpt = torch.load(args.from_file, map_location='cpu')
            if 'opt' in ckpt:
                # 只加载优化器状态，如果参数数量匹配
                if len(ckpt['opt']['param_groups'][0]['params']) == len(opt.param_groups[0]['params']):
                    opt.load_state_dict(ckpt['opt'])
                    print(colored("优化器状态已从检查点加载。", "green"))
                    # 调整调度器步骤
                    print(f"将调度器推进到步骤 {start_step}...")
                    sched.step_ = 0  # 重置内部步骤
                    # 正确推进调度器
                    target_lr_ratio = min(start_step / cfg.warmup_steps, 0.5 * (1 + math.cos(math.pi * (start_step - cfg.warmup_steps) / (cfg.total_steps - cfg.warmup_steps)))) if start_step > cfg.warmup_steps else start_step / cfg.warmup_steps
                    target_lr_ratio = max(0, target_lr_ratio)  # 确保比率非负

                    # 设置当前步骤并更新优化器中的学习率
                    sched.step_ = start_step
                    for g in opt.param_groups:
                        g['lr'] = cfg.lr * target_lr_ratio
                    print(colored(f"调度器步骤设置为 {sched.step_}，学习率设置为 {opt.param_groups[0]['lr']:.2e}",
                                  "green"))
                else:
                    print(colored("警告：检查点中优化器参数数量与当前参数不匹配。使用新的优化器状态。", "yellow"))
            else:
                print(colored("警告：检查点中未找到优化器状态。将使用新的优化器状态。", "yellow"))
        except Exception as e:
            print(colored(f"加载优化器状态失败: {e}. 将使用新的优化器状态。", "red"))

    # 梯度累积
    grad_accum_steps = cfg.grad_accum_steps
    print(colored(f"使用梯度累积，累积步数: {grad_accum_steps}", "cyan"))
    effective_batch_size = cfg.batch_size * grad_accum_steps
    print(colored(f"每种任务类型的有效批次大小: {effective_batch_size}", "cyan"))

    # 加载数据集
    print(colored("\n=== 加载数据集 ===", "cyan"))
    deletion_dir = os.path.join(args.data_dir, "deletion")
    insertion_dir = os.path.join(args.data_dir, "insertion")

    try:
        # 加载数据集
        task_datasets = {}
        task_loaders = {}
        task_iters = {}

        # 如果不跳过预测任务
        if not cfg.skip_prediction:
            print("加载预测任务数据集...")
            prediction_dataset = PredictionTaskDataset(args.data_dir, "train", cfg.seq_len, shuffle=True)
            task_datasets["prediction"] = prediction_dataset
            task_loaders["prediction"] = DataLoader(
                prediction_dataset, batch_size=cfg.batch_size, shuffle=True,
                num_workers=2, pin_memory=True, collate_fn=collate_fn, drop_last=True
            )
            task_iters["prediction"] = iter(task_loaders["prediction"])

        # 始终加载删除和插入任务
        print("加载删除任务数据集...")
        deletion_dataset = DeletionTaskDataset(deletion_dir, "train", shuffle=True, del_token_id=del_token_id)
        task_datasets["deletion"] = deletion_dataset
        task_loaders["deletion"] = DataLoader(
            deletion_dataset, batch_size=cfg.batch_size, shuffle=True,
            num_workers=2, pin_memory=True, collate_fn=collate_fn, drop_last=True
        )
        task_iters["deletion"] = iter(task_loaders["deletion"])

        print("加载插入任务数据集...")
        insertion_dataset = InsertionTaskDataset(insertion_dir, "train", shuffle=True)
        task_datasets["insertion"] = insertion_dataset
        task_loaders["insertion"] = DataLoader(
            insertion_dataset, batch_size=cfg.batch_size, shuffle=True,
            num_workers=2, pin_memory=True, collate_fn=collate_fn, drop_last=True
        )
        task_iters["insertion"] = iter(task_loaders["insertion"])

    except FileNotFoundError as e:
        print(colored(f"加载数据集时出错: {e}", "red"))
        print("请确保预处理已完成，所有数据目录/文件都存在。")
        print(f"检查路径: {deletion_dir}, {insertion_dir}, {args.data_dir}/*_prediction_inputs.pt")
        return

    # 检查数据集是否为空
    for task_name, dataset in task_datasets.items():
        if len(dataset) == 0:
            print(colored(f"错误: {task_name} 数据集为空", "red"))
            print("无法继续训练。")
            return

    # 打印数据集大小
    dataset_sizes = {task: len(dataset) for task, dataset in task_datasets.items()}
    print(colored(f"数据集大小: {dataset_sizes}", "green"))

    # 主训练循环
    print(colored(f"\n=== 开始训练: 从步骤 {start_step + 1} 到 {cfg.total_steps} ===", "green"))
    model.train()  # 设置模型为训练模式
    if start_step == 0:
        opt.zero_grad()  # 如果不是恢复训练，清零梯度

    # 任务顺序（如果跳过预测则只有两个任务）
    tasks = ["deletion", "insertion"]
    if not cfg.skip_prediction:
        tasks.append("prediction")

    # 创建进度条
    pbar = tqdm.trange(start_step + 1, cfg.total_steps + 1, desc="训练步骤")

    # 训练循环
    for step in pbar:
        # 轮流选择任务
        task_idx = (step - 1) % len(tasks)
        task_type = tasks[task_idx]

        # 获取当前任务的数据加载器和迭代器
        loader = task_loaders[task_type]
        loader_iter = task_iters[task_type]

        # 获取下一批数据，处理迭代器耗尽的情况
        batch, new_iter = get_next_batch(loader_iter, loader, task_type, step)
        task_iters[task_type] = new_iter  # 更新迭代器

        # 将数据移至设备
        x = batch['sequences'].to(device)
        target_indices = batch['indices'].to(device)
        target_tokens = batch['tokens'].to(device)

        # 使用自动混合精度进行前向传播
        with torch.cuda.amp.autocast(enabled=cfg.fp16):
            # 前向传播，model内部会根据target_index决定使用lm_head还是edit_head逻辑
            out = model(
                input_ids=x,
                target_index=target_indices,
                target_token=target_tokens
            )
            # 损失来自index_loss和token_loss
            loss = out["loss"] / grad_accum_steps  # 对梯度累积进行归一化

        # 反向传播和梯度累积
        scaled_loss = scaler.scale(loss)
        scaled_loss.backward()

        # 梯度累积完成后进行优化器步骤
        if step % grad_accum_steps == 0:
            # 缩放并裁剪可训练参数的梯度
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),  # 只裁剪可训练参数
                cfg.grad_clip
            )
            # 优化器步骤只更新可训练参数
            scaler.step(opt)
            scaler.update()
            # 基于优化器步骤调整调度器
            sched.step()
            opt.zero_grad(set_to_none=True)  # 使用set_to_none=True减少内存占用

        # 日志记录和检查点保存
        pbar.set_postfix(
            task=task_type,
            loss=f"{loss.item() * grad_accum_steps:.3f}",  # 记录未归一化的损失
            idx_loss=f"{out.get('idx_loss', torch.tensor(0.0)).item():.3f}",
            tok_loss=f"{out.get('tok_loss', torch.tensor(0.0)).item():.3f}",
            lr=f"{opt.param_groups[0]['lr']:.2e}"
        )

        # 保存检查点
        if step % cfg.ckpt_every == 0 and step > 0:
            ckpt_path = f"{args.outdir}/sft_step{step}.pt"
            save_ckpt(model, opt, step, ckpt_path, tokenizer=tokenizer)
            print(colored(f"在步骤 {step} 保存检查点: {ckpt_path}", "green"))

    # 保存最终模型
    final_ckpt_path = f"{args.outdir}/sft_final_{cfg.total_steps}.pt"
    save_ckpt(model, opt, cfg.total_steps, final_ckpt_path, tokenizer=tokenizer)
    print(colored(f"\n训练完成。最终模型保存到: {final_ckpt_path}", "green"))


if __name__ == "__main__":
    main()
