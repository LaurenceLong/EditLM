# src/train_sft_hf.py
"""
使用预处理的二进制数据快速训练EditLM模型
明确区分训练阶段：
1. 预测任务阶段（可选）：仅训练backbone的LM Head，用于基础语言建模能力。
2. 编辑任务阶段：仅训练新增的编辑相关头部（index_head, edit_head, projections），
   冻结整个backbone。交替进行删除和插入任务。
"""
import argparse
import os
import warnings  # Import warnings

import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader

from data import collate_fn, DeletionTaskDataset, InsertionTaskDataset
from model_hf import EditLMHF
from tokenizer import get_tokenizer, get_del_token_id
from utils import save_ckpt, WarmupCosine, load_model_from_ckpt


def set_trainable_parts(model: EditLMHF, phase: str):
    """
    根据训练阶段设置模型的可训练部分。

    Args:
        model: The EditLMHF model.
        phase: The current training phase ('prediction' or 'editing').
    """
    # 1. 首先冻结所有参数
    for param in model.parameters():
        param.requires_grad_(False)

    if phase == 'prediction':
        print("Setting trainable parts for PREDICTION phase (only LM head)")
        # 2. 仅解冻 LM head
        try:
            lm_head = None
            # 尝试常见的LM Head属性名
            if hasattr(model.backbone, 'lm_head') and model.backbone.lm_head is not None:
                lm_head = model.backbone.lm_head
                print("Found LM head: model.backbone.lm_head")
            elif hasattr(model.backbone, 'output_embedding') and model.backbone.output_embedding is not None:  # 备用方案
                lm_head = model.backbone.output_embedding
                print("Found LM head: model.backbone.output_embedding")

            if lm_head is not None:
                for param in lm_head.parameters():
                    param.requires_grad_(True)
                print(
                    f"Successfully unfroze {sum(p.numel() for p in lm_head.parameters() if p.requires_grad)} parameters in LM head.")
            else:
                # Fallback: 尝试解冻最后一个模块（通常是LM Head）
                # 注意：这种方法比较脆弱，可能不适用于所有模型结构
                modules = list(model.backbone.modules())
                if modules:
                    last_module = modules[-1]
                    # 检查是否是线性层且与嵌入层权重不同（避免误判输入嵌入层）
                    input_embed_weight = model.backbone.get_input_embeddings().weight
                    if isinstance(last_module, nn.Linear) and id(getattr(last_module, 'weight', None)) != id(
                            input_embed_weight):
                        warnings.warn(
                            f"Attempting to unfreeze last module '{type(last_module).__name__}' as LM head (fallback).")
                        for param in last_module.parameters():
                            param.requires_grad_(True)
                        print(
                            f"Unfroze {sum(p.numel() for p in last_module.parameters() if p.requires_grad)} parameters in the last module.")
                    else:
                        warnings.warn(
                            "Could not reliably identify and unfreeze LM head. Prediction phase might not train correctly.")
                else:
                    warnings.warn("Could not find any modules in backbone. LM head unfreezing failed.")

        except AttributeError as e:
            warnings.warn(f"Error accessing potential LM head - {e}. Prediction phase might not train correctly.")

    elif phase == 'editing':
        print("Setting trainable parts for EDITING phase (only edit heads/projections)")
        # 2. 仅解冻编辑相关的头部和投影层
        trainable_components = [
            model.index_head, model.edit_head,  # 注意：edit_head 可能与 input_embed 共享权重，但仍需解冻以接收梯度
            model.triple_proj, model.fuse_proj
        ]
        # 添加可学习的 embeddings
        if hasattr(model, 'gap_token_embed'):
            trainable_components.append(model.gap_token_embed)  # nn.Parameter 直接添加
        if hasattr(model, 'boundary_embed'):
            trainable_components.append(model.boundary_embed)  # nn.Parameter 直接添加

        unfrozen_count = 0
        component_names = ["index_head", "edit_head", "triple_proj", "fuse_proj", "gap_token_embed", "boundary_embed"]
        for i, component in enumerate(trainable_components):
            if component is not None:
                # Handle both nn.Module and nn.Parameter
                if isinstance(component, nn.Module):
                    for param in component.parameters():
                        param.requires_grad_(True)
                        unfrozen_count += param.numel()
                elif isinstance(component, nn.Parameter):
                    component.requires_grad_(True)
                    unfrozen_count += component.numel()
                print(f"Unfroze {component_names[i]}")

        print(f"Successfully unfroze {unfrozen_count} parameters for editing heads/projections.")

    else:
        warnings.warn(f"Unknown training phase '{phase}'. No parameters set to trainable.")

    # 打印最终的可训练参数数量进行验证
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Phase '{phase}': {trainable_params / 1e6:.2f}M trainable parameters out of {total_params / 1e6:.2f}M total parameters.")


def main():
    ap = argparse.ArgumentParser(
        description="Train EditLM with separate phases for prediction and editing.")  # Added description
    ap.add_argument("--outdir", required=True, help="模型保存目录")
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B",
                    help="HF checkpoint, e.g. Qwen/Qwen2.5-0.5B")
    ap.add_argument("--from_file", help="Load from ckpt file path (resumes training state)")
    # Removed --freeze argument
    # ap.add_argument("--freeze", action="store_true",
    #                 help="若指定，仅训练三个新头 (triple_proj / index_head / token_head)")
    ap.add_argument("--data_dir", default="./data/wikitext_processed",
                    help="预处理数据目录")
    ap.add_argument("--batch_size", type=int, default=8, help="训练批次大小")
    ap.add_argument("--steps", type=int, default=100000, help="编辑任务训练总步数")
    ap.add_argument("--skip_prediction", action="store_true", default=False,
                    # Changed from --train_prediction and default
                    help="若指定，跳过预测任务的训练阶段")
    ap.add_argument("--prediction_epochs", type=int, default=1,
                    help="预测任务训练的epoch数 (如果未跳过)")
    args = ap.parse_args()

    # 创建输出目录
    os.makedirs(args.outdir, exist_ok=True)

    class _Cfg:
        batch_size = args.batch_size
        total_steps = args.steps  # Refers to editing steps
        lr = 2e-4
        warmup_steps = 1_000
        grad_clip = 1.0
        fp16 = True  # Assumes CUDA is available, adjust if needed
        ckpt_every = 5_000  # Checkpoint during editing phase

    cfg = _Cfg()

    # 加载元数据获取序列长度
    metadata_path = os.path.join(args.data_dir, "train_metadata.pt")
    try:
        if not os.path.exists(metadata_path):
            # Try looking inside deletion/insertion dirs if top-level doesn't exist
            metadata_path_alt1 = os.path.join(args.data_dir, "deletion", "train_metadata.pt")
            metadata_path_alt2 = os.path.join(args.data_dir, "insertion", "train_metadata.pt")
            if os.path.exists(metadata_path_alt1):
                metadata_path = metadata_path_alt1
            elif os.path.exists(metadata_path_alt2):
                metadata_path = metadata_path_alt2
            else:
                raise FileNotFoundError("Metadata file not found in data_dir or subdirectories.")
        metadata = torch.load(metadata_path)
        cfg.seq_len = metadata["seq_len"]
        print(f"Loaded sequence length {cfg.seq_len} from metadata: {metadata_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure preprocessing is complete and data_dir is correct.")
        return
    except KeyError:
        print(f"Error: 'seq_len' not found in metadata file: {metadata_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.fp16 = cfg.fp16 and (device == "cuda")  # Enable fp16 only on CUDA
    print(f"使用设备: {device}, FP16 enabled: {cfg.fp16}")

    # --- Model and Tokenizer Loading ---
    print(f"加载模型: {args.base}")
    tokenizer = None
    model = None
    start_step = 0  # For resuming training

    if args.from_file:
        print(f"加载检查点: {args.from_file}")
        try:
            model, tokenizer, start_step = load_model_from_ckpt(args.from_file, base_model=args.base)
            model.to(device)
            print(f"模型从检查点加载完成，将从步骤 {start_step} 继续编辑任务训练。")
            # If loading from checkpoint, usually skip prediction phase unless explicitly told otherwise
            if not args.skip_prediction and start_step == 0:
                print(
                    "Checkpoint loaded but start_step is 0, will run prediction phase unless --skip_prediction is set.")
            elif start_step > 0:
                print("Resuming from checkpoint, skipping prediction phase.")
                args.skip_prediction = True  # Force skip prediction when resuming
        except Exception as e:
            print(f"从检查点加载失败: {e}. 将尝试从基础模型重新开始。")
            args.from_file = None  # Reset from_file to load fresh model
            start_step = 0

    if model is None:  # If not loaded from checkpoint or loading failed
        # Load tokenizer fresh
        print("加载新的 Tokenizer...")
        tokenizer = get_tokenizer(args.base, use_fast=True)

        model = EditLMHF(base_model=args.base, index_loss_weight=1.0).to(device)
    del_token_id = get_del_token_id(tokenizer)

    # --- Optimizer and Scheduler ---
    # Optimizer needs to be aware of ALL parameters that *might* be trained across phases.
    # The requires_grad flag will control which ones actually get gradients.
    print("创建优化器和调度器...")
    opt = torch.optim.AdamW(
        model.parameters(),  # Pass all parameters initially
        lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    # Scheduler total steps should correspond to the editing phase steps
    sched = WarmupCosine(opt, cfg, cfg.total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)

    # Load optimizer state if resuming
    if args.from_file and start_step > 0:
        try:
            ckpt = torch.load(args.from_file, map_location='cpu')
            if 'opt' in ckpt:
                opt.load_state_dict(ckpt['opt'])
                print("优化器状态已从检查点加载。")
                # Adjust scheduler step to match resumed step
                # Simple approach: step scheduler forward 'start_step' times
                print(f"Advancing scheduler to step {start_step}...")
                # Reset internal step and fast-forward (more robust than just setting step_)
                sched.step_ = 0
                for _ in range(start_step):
                    sched.step()
                print(f"Scheduler step is now {sched.step_}")

            else:
                print("警告：检查点中未找到优化器状态。将使用新的优化器状态。")
        except Exception as e:
            print(f"加载优化器状态失败: {e}. 将使用新的优化器状态。")

    # Gradient Accumulation
    grad_accum_steps = 4  # You might want to make this an argument
    print(f"使用梯度累积，累积步数: {grad_accum_steps}")
    effective_batch_size = cfg.batch_size * grad_accum_steps
    print(f"有效批次大小: {effective_batch_size}")

    # === Phase 1: Prediction Task Training (Optional) ===
    if not args.skip_prediction:
        print("\n=== Phase 1: Prediction Task Training (LM Head Only) ===")
        set_trainable_parts(model, 'prediction')  # Configure model for prediction training

        # Load prediction task data
        prediction_data_path = os.path.join(args.data_dir, "train_prediction_inputs.pt")
        if not os.path.exists(prediction_data_path):
            print(f"错误：预测任务数据未找到: {prediction_data_path}")
            print("请确保预处理已完成。跳过预测任务训练。")
        else:
            print(f"加载预测任务数据: {prediction_data_path}")
            prediction_data = torch.load(prediction_data_path)
            total_sequences = len(prediction_data)
            if total_sequences < cfg.batch_size:
                print(f"警告：预测数据样本 ({total_sequences}) 少于批次大小 ({cfg.batch_size})。跳过预测训练。")
            else:
                total_batches_per_epoch = total_sequences // cfg.batch_size
                print(f"预测任务训练 {args.prediction_epochs} 个epoch，每个epoch {total_batches_per_epoch} 批次")

                model.train()  # Set model to train mode
                opt.zero_grad()  # Ensure grads are zeroed before starting

                global_pred_step = 0
                for epoch in range(args.prediction_epochs):
                    print(f"Prediction Epoch {epoch + 1}/{args.prediction_epochs}")
                    indices = torch.randperm(total_sequences)  # Shuffle data each epoch
                    pbar_pred = tqdm.tqdm(range(total_batches_per_epoch), desc=f"Pred Epoch {epoch + 1}")

                    for batch_idx in pbar_pred:
                        start_idx = batch_idx * cfg.batch_size
                        end_idx = start_idx + cfg.batch_size
                        batch_indices = indices[start_idx:end_idx]

                        x = prediction_data[batch_indices].to(device)  # [B, L]

                        # Target for prediction task (as per original logic):
                        # Predict the *last* token using the representation *before* it.
                        # The model's forward pass handles this when target_index == seq_len.
                        target_index = torch.full((cfg.batch_size,), fill_value=cfg.seq_len, device=device,
                                                  dtype=torch.long)
                        target_token = x[:, -1]  # The actual last token is the target

                        with torch.amp.autocast(device, enabled=cfg.fp16):
                            # Only LM head should be active and produce gradients
                            out = model(
                                input_ids=x,
                                target_index=target_index,
                                target_token=target_token
                            )
                            # Loss should primarily come from tok_loss driven by lm_head
                            # idx_loss might be computed but won't affect lm_head grads
                            loss = out["loss"] / grad_accum_steps

                        # Accumulate gradients
                        scaler.scale(loss).backward()
                        global_pred_step += 1

                        if global_pred_step % grad_accum_steps == 0:
                            # Unscale and clip gradients for TRAINABLE parameters only
                            scaler.unscale_(opt)
                            torch.nn.utils.clip_grad_norm_(
                                (p for p in model.parameters() if p.requires_grad),  # Only clip trainable
                                cfg.grad_clip
                            )
                            # Optimizer step applies updates ONLY to trainable parameters
                            scaler.step(opt)
                            scaler.update()
                            opt.zero_grad()
                            # Note: We do NOT step the scheduler here, as it's based on editing steps

                        pbar_pred.set_postfix(loss=f"{loss.item() * grad_accum_steps:.3f}")

                # Save model after prediction phase
                pred_ckpt_path = f"{args.outdir}/prediction_phase_final.pt"
                save_ckpt(model, opt, 0, pred_ckpt_path, tokenizer=tokenizer)  # Save with step 0
                print(f"预测任务训练完成，模型已保存至: {pred_ckpt_path}")

    else:
        print("\nSkipping Phase 1: Prediction Task Training")

    # === Phase 2: Editing Task Training (Edit Heads Only) ===
    print("\n=== Phase 2: Editing Task Training (Edit Heads Only) ===")
    set_trainable_parts(model, 'editing')  # Configure model for editing training

    # Load deletion and insertion task datasets
    deletion_dir = os.path.join(args.data_dir, "deletion")
    insertion_dir = os.path.join(args.data_dir, "insertion")

    try:
        deletion_dataset = DeletionTaskDataset(deletion_dir, "train", shuffle=True, del_token_id=del_token_id)
        insertion_dataset = InsertionTaskDataset(insertion_dir, "train", shuffle=True)
    except FileNotFoundError as e:
        print(f"Error loading editing datasets: {e}")
        print("Please ensure preprocessing is complete and data directories exist.")
        print(f"Checked: {deletion_dir}, {insertion_dir}")
        return

    # Check if datasets are empty
    if len(deletion_dataset) == 0 or len(insertion_dataset) == 0:
        print("Error: Deletion or Insertion dataset is empty. Cannot proceed with editing training.")
        return

    # Create data loaders
    # shuffle=False because datasets handle shuffling internally via chunk loading order
    deletion_loader = DataLoader(
        deletion_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=collate_fn, drop_last=True  # drop_last is important
    )
    insertion_loader = DataLoader(
        insertion_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=collate_fn, drop_last=True  # drop_last is important
    )

    # Create iterators
    deletion_iter = iter(deletion_loader)
    insertion_iter = iter(insertion_loader)

    print(f"开始编辑任务训练，从步骤 {start_step + 1} 到 {cfg.total_steps}")
    model.train()  # Set model to train mode
    if start_step == 0:  # Zero grad only if not resuming
        opt.zero_grad()

    pbar_edit = tqdm.trange(start_step + 1, cfg.total_steps + 1, desc="Editing Task")

    for step in pbar_edit:
        # Alternate between deletion and insertion tasks
        task_type = "deletion" if step % 2 == 0 else "insertion"
        loader_iter = deletion_iter if task_type == "deletion" else insertion_iter
        loader = deletion_loader if task_type == "deletion" else insertion_loader

        try:
            batch = next(loader_iter)
        except StopIteration:
            # Reset iterator if exhausted
            loader_iter = iter(loader)
            batch = next(loader_iter)
            # Update the main iterator variable as well
            if task_type == "deletion":
                deletion_iter = loader_iter
            else:
                insertion_iter = loader_iter

        # Move data to device
        x = batch['sequences'].to(device)
        target_indices = batch['indices'].to(device)
        target_tokens = batch['tokens'].to(device)

        # Forward pass with autocast
        with torch.amp.autocast(device, enabled=cfg.fp16):
            # Only edit heads should be active and produce gradients
            out = model(
                input_ids=x,
                target_index=target_indices,
                target_token=target_tokens
            )
            # Loss comes from index_loss and token_loss (from edit_head)
            loss = out["loss"] / grad_accum_steps

        # Backward pass and gradient accumulation
        scaler.scale(loss).backward()

        if step % grad_accum_steps == 0:
            # Unscale and clip gradients for TRAINABLE parameters only
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),  # Only clip trainable
                cfg.grad_clip
            )
            # Optimizer step applies updates ONLY to trainable parameters (edit heads)
            scaler.step(opt)
            scaler.update()
            # Step the scheduler based on optimizer steps
            sched.step()
            opt.zero_grad()

        # Logging and Checkpointing
        pbar_edit.set_postfix(task=task_type, loss=f"{loss.item() * grad_accum_steps:.3f}",
                              lr=f"{sched.opt.param_groups[0]['lr']:.2e}")

        if step % cfg.ckpt_every == 0:
            ckpt_path = f"{args.outdir}/sft_step{step}.pt"
            save_ckpt(model, opt, step, ckpt_path, tokenizer=tokenizer)

    # Save final model
    final_ckpt_path = f"{args.outdir}/sft_final_{cfg.total_steps}.pt"
    save_ckpt(model, opt, cfg.total_steps, final_ckpt_path, tokenizer=tokenizer)
    print(f"编辑任务训练完成，最终模型已保存至: {final_ckpt_path}")


if __name__ == "__main__":
    main()
