"""
预训练脚本，使用文本shift生成的数据进行预训练
"""
import argparse
import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb

from model.modeling import EditLM
from model.tokenizer import get_tokenizer
from data.pretraining import TextShiftDataset, PretrainingDataset, pretrain_collate_fn
from config.config import Config

def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Pretrain Index-Token Editor model")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to raw text corpus for pretraining")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run pretraining on")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for pretraining")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of pretraining epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    args = parser.parse_args()

    # 加载配置
    config = Config()

    # 更新配置
    config.output_dir = args.output_dir
    config.batch_size = args.batch_size
    config.num_train_epochs = args.num_epochs
    config.learning_rate = args.lr

    # 添加语言模型损失权重
    config.lm_loss_weight = 0.5  # 语言模型预测损失的权重

    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)

    # 设置随机种子
    set_seed(config.seed)

    # 加载词元器
    tokenizer = get_tokenizer(config)

    # 初始化模型
    model = EditLM(
        config=config,
        model_name_or_path=config.model_name_or_path,
        tokenizer_length=len(tokenizer)
    )

    # 将模型移至指定设备
    model.to(args.device)

    # 准备预训练数据
    print("Generating pretraining data...")
    text_shift_dataset = TextShiftDataset(
        corpus_path=args.corpus_path,
        tokenizer=tokenizer,
        config=config,
        max_samples=args.max_samples
    )

    print(f"Generated {len(text_shift_dataset)} text shift samples")

    # 处理为训练样本
    pretraining_dataset = PretrainingDataset(
        samples=text_shift_dataset.samples,
        tokenizer=tokenizer,
        config=config
    )

    print(f"Processed into {len(pretraining_dataset)} training samples")

    # 创建数据加载器
    dataloader = DataLoader(
        pretraining_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: pretrain_collate_fn(batch, tokenizer)
    )

    # 计算训练步数
    total_steps = len(dataloader) * config.num_train_epochs

    # 设置优化器和学习率调度器
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps
    )

    # 初始化wandb
    wandb.init(project="index-token-editor-pretraining", config=vars(config))

    # 预训练循环
    global_step = 0

    for epoch in range(config.num_train_epochs):
        model.train()
        epoch_loss = 0

        with tqdm(dataloader, desc=f"Pretraining Epoch {epoch+1}/{config.num_train_epochs}") as t:
            for batch in t:
                # 将数据移动到设备
                input_ids = batch["input_ids"].to(args.device)
                attention_mask = batch["attention_mask"].to(args.device)
                target_token_ids = batch["target_token_ids"].to(args.device)
                target_index_positions = batch["target_index_positions"].to(args.device)
                lm_targets = batch["lm_targets"].to(args.device)
                lm_positions = batch["lm_positions"].to(args.device)

                # 前向传播
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    target_token_ids=target_token_ids,
                    target_index_positions=target_index_positions,
                    lm_targets=lm_targets,
                    lm_positions=lm_positions,
                    return_dict=True
                )

                loss = outputs["loss"]
                token_loss = outputs["token_loss"]
                index_loss = outputs["index_loss"]
                lm_loss = outputs.get("lm_loss", torch.tensor(0.0).to(args.device))

                # 处理梯度累积
                loss = loss / config.gradient_accumulation_steps
                loss.backward()

                epoch_loss += loss.item()

                # 更新进度条
                t.set_postfix(loss=loss.item(), token_loss=token_loss.item(),
                             index_loss=index_loss.item(), lm_loss=lm_loss.item())

                # 梯度累积
                if (global_step + 1) % config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                global_step += 1

                # 日志记录
                if global_step % config.logging_steps == 0:
                    wandb.log({
                        "loss": loss.item() * config.gradient_accumulation_steps,
                        "token_loss": token_loss.item(),
                        "index_loss": index_loss.item(),
                        "lm_loss": lm_loss.item(),
                        "learning_rate": scheduler.get_last_lr()[0],
                        "step": global_step
                    })

                # 保存检查点
                if global_step % config.save_steps == 0:
                    output_dir = os.path.join(config.output_dir, f"pretrain-checkpoint-{global_step}")
                    os.makedirs(output_dir, exist_ok=True)

                    # 保存模型
                    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))

                    # 保存tokenizer
                    tokenizer.save_pretrained(output_dir)

        # 每个epoch结束后保存模型
        output_dir = os.path.join(config.output_dir, f"pretrain-epoch-{epoch+1}")
        os.makedirs(output_dir, exist_ok=True)

        # 保存模型
        torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))

        # 保存tokenizer
        tokenizer.save_pretrained(output_dir)

        # 记录每个epoch的损失
        wandb.log({
            "epoch": epoch + 1,
            "epoch_loss": epoch_loss / len(dataloader),
            "step": global_step
        })

    # 保存最终模型
    output_dir = os.path.join(config.output_dir, "pretrained-final")
    os.makedirs(output_dir, exist_ok=True)

    # 保存模型
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))

    # 保存tokenizer
    tokenizer.save_pretrained(output_dir)

    wandb.finish()

    print(f"Pretraining completed! Model saved to {output_dir}")

if __name__ == "__main__":
    main()
