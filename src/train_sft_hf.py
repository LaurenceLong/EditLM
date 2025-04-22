"""
使用预处理的二进制数据快速训练EditLM模型
明确区分三种任务类型：
1. 预测任务：使用高效的token shift方式一次性训练
2. 删除任务：识别并删除错误或多余的token
3. 插入任务：在正确位置插入缺失的token

任务1仅训练一次，任务2和3交替进行，确保模型全面学习各种编辑能力
"""
import argparse
import os

import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader

from data import collate_fn, DeletionTaskDataset, InsertionTaskDataset
from model_hf import EditLMHF
from tokenizer import get_tokenizer, NUM_ADDED_TOKENS, get_del_token_id
from utils import save_ckpt, WarmupCosine, load_model_from_ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="模型保存目录")
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B",
                    help="HF checkpoint, e.g. Qwen/Qwen2.5-0.5B")
    ap.add_argument("--from_file", help="Load from ckpt file path")
    ap.add_argument("--freeze", action="store_true",
                    help="若指定，仅训练三个新头 (triple_proj / index_head / token_head)")
    ap.add_argument("--data_dir", default="./data/wikitext_processed",
                    help="预处理数据目录")
    ap.add_argument("--batch_size", type=int, default=8, help="训练批次大小")
    ap.add_argument("--steps", type=int, default=100000, help="训练步数")
    ap.add_argument("--skip_prediction", action="store_true",
                    help="若指定，跳过预测任务的训练")
    ap.add_argument("--prediction_epochs", type=int, default=1,
                    help="预测任务训练的epoch数")
    args = ap.parse_args()

    # 创建输出目录
    os.makedirs(args.outdir, exist_ok=True)


    class _Cfg:
        batch_size   = args.batch_size
        total_steps  = args.steps
        lr           = 2e-4
        warmup_steps = 1_000
        grad_clip    = 1.0
        fp16         = True
        ckpt_every   = 5_000
    cfg = _Cfg()

    # 加载元数据获取序列长度
    metadata_path = os.path.join(args.data_dir, "train_metadata.pt")
    metadata = torch.load(metadata_path)
    cfg.seq_len = metadata["seq_len"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    print(f"加载模型: {args.base}")
    tokenizer = None
    num_added_toks = 0
    if args.from_file:
        model, tokenizer, _  = load_model_from_ckpt(args.from_file, base_model=args.base)
        model.to(device)
    else:
        model = EditLMHF(base_model=args.base, index_loss_weight=1.0).to(device)

    if tokenizer is None:
        # 加载tokenizer
        tokenizer = get_tokenizer(args.base, use_fast=True)
        # 添加<DEL>特殊标记
        num_added_toks = NUM_ADDED_TOKENS[0]

    del_token_id = get_del_token_id(tokenizer)

    # 如果使用预训练模型，需要调整模型的嵌入大小以匹配新的词汇表大小
    if num_added_toks > 0:
        model.backbone.resize_token_embeddings(len(tokenizer))
        # 同时也需要调整token_head的大小
        model.token_head = nn.Linear(model.hidden_size, len(tokenizer), bias=False)
        # 重新绑定共享权重
        model.token_head.weight = model.backbone.get_input_embeddings().weight
        print(f"已调整模型词嵌入大小为: {len(tokenizer)}")

    if args.freeze:
        print("冻结backbone参数，只训练新增头部")
        for p in model.backbone.parameters():
            p.requires_grad_(False)

    # 计算可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数数量: {trainable_params/1e6:.2f}M")

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    sched = WarmupCosine(opt, cfg, cfg.total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.fp16)  # 新

    # 使用梯度累积减少内存需求
    grad_accum_steps = 4
    print(f"使用梯度累积，累积步数: {grad_accum_steps}")

    # 第一阶段：预测任务训练（只训练一次）
    if not args.skip_prediction:
        print("\n=== 第一阶段：预测任务训练 ===")

        # 加载预测任务数据
        print(f"加载预测任务数据...")
        prediction_data = torch.load(os.path.join(args.data_dir, "train_prediction_inputs.pt"))

        # 计算总批次数
        total_sequences = len(prediction_data)
        total_batches = total_sequences // cfg.batch_size

        # 设置预测任务的训练轮次
        print(f"预测任务训练 {args.prediction_epochs} 个epoch")

        for epoch in range(args.prediction_epochs):
            print(f"Epoch {epoch+1}/{args.prediction_epochs}")

            # 打乱数据
            indices = torch.randperm(total_sequences)

            # 创建进度条
            pbar = tqdm.tqdm(range(total_batches), desc="预测任务训练")

            # 对每个批次进行训练
            for batch_idx in pbar:
                # 获取当前批次的数据
                start_idx = batch_idx * cfg.batch_size
                end_idx = min((batch_idx + 1) * cfg.batch_size, total_sequences)

                # 如果最后一个批次大小不足，则跳过
                if end_idx - start_idx < cfg.batch_size:
                    continue

                # 获取当前批次的索引
                batch_indices = indices[start_idx:end_idx]

                # 提取输入序列
                x = prediction_data[batch_indices].to(device)

                # Teacher forcing：把最后一个gap当作gold
                target_index = torch.full((cfg.batch_size,), fill_value=cfg.seq_len, device=device)
                target_token = x[:, -1]

                with torch.amp.autocast('cuda', enabled=cfg.fp16):  # 新
                    out = model(
                        input_ids=x,
                        target_index=target_index,
                        target_token=target_token
                    )
                    loss = out["loss"] / grad_accum_steps

                scaler.scale(loss).backward()

                # 梯度累积：每累积指定步数才更新一次
                if (batch_idx + 1) % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        (p for p in model.parameters() if p.requires_grad),
                        cfg.grad_clip
                    )
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()
                    sched.step()

                pbar.set_description(f"预测任务 loss {loss.item() * grad_accum_steps:.3f}")

        # 保存预测任务训练后的模型
        save_ckpt(model, opt, 0, f"{args.outdir}/prediction_phase.pt")
        print("预测任务训练完成，模型已保存")

    # 第二阶段：删除和插入任务交替训练
    print("\n=== 第二阶段：编辑任务训练 ===")


    deletion_dir = os.path.join(args.data_dir, "deletion")
    insertion_dir = os.path.join(args.data_dir, "insertion")
    # 加载删除和插入任务数据集
    deletion_dataset = DeletionTaskDataset(deletion_dir, "train", shuffle=True, del_token_id=del_token_id)
    insertion_dataset = InsertionTaskDataset(insertion_dir, "train", shuffle=True)

    # 创建数据加载器
    deletion_loader = DataLoader(
        deletion_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,  # 数据集内部已经实现了shuffle
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn
    )

    insertion_loader = DataLoader(
        insertion_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,  # 数据集内部已经实现了shuffle
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # 创建迭代器
    deletion_iter = iter(deletion_loader)
    insertion_iter = iter(insertion_loader)

    # 交替训练删除和插入任务
    pbar = tqdm.trange(1, cfg.total_steps + 1)
    for step in pbar:
        # 交替选择任务
        if step % 2 == 0:  # 偶数步：删除任务
            try:
                batch = next(deletion_iter)
            except StopIteration:
                deletion_iter = iter(deletion_loader)
                batch = next(deletion_iter)

            task_name = "删除"
        else:  # 奇数步：插入任务
            try:
                batch = next(insertion_iter)
            except StopIteration:
                insertion_iter = iter(insertion_loader)
                batch = next(insertion_iter)

            task_name = "插入"

        # 将数据移到设备上
        x = batch['sequences'].to(device)
        target_indices = batch['indices'].to(device)
        target_tokens = batch['tokens'].to(device)

        # 确保batch大小正确
        if x.size(0) < cfg.batch_size:
            continue

        with torch.amp.autocast('cuda', enabled=cfg.fp16):  # 新
            out = model(
                input_ids=x,
                target_index=target_indices,
                target_token=target_tokens
            )
            loss = out["loss"] / grad_accum_steps

        scaler.scale(loss).backward()

        # 梯度累积：每累积指定步数才更新一次
        if step % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                cfg.grad_clip
            )
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            sched.step()

        pbar.set_description(f"{task_name}任务 loss {loss.item() * grad_accum_steps:.3f}")

        # 定期保存模型
        if step % cfg.ckpt_every == 0:
            save_ckpt(model, opt, step, f"{args.outdir}/sft_step{step}.pt")

    # 保存最终模型
    save_ckpt(model, opt, args.total_steps, f"{args.outdir}/sft_final_{cfg.total_steps}.pt")
    print("训练完成，最终模型已保存")


if __name__ == "__main__":
    main()
