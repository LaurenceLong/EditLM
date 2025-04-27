"""
预处理WikiText-103数据集，将其转换为二进制格式以加速训练。
优化版本：使用流式处理和直接写入磁盘，避免内存占用
区分三种任务类型，并保持1:1:1的比例：
1. 预测任务（标准自回归，独立处理）
2. 删除任务（在序列中插入随机token，训练模型识别删除）
3. 插入任务（从序列中删除token，训练模型学习插入）
"""
import argparse
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from tokenizer import get_tokenizer, get_del_token_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-0.5B", help="模型名称，用于加载tokenizer")
    parser.add_argument("--seq_len", type=int, default=256, help="序列长度")
    parser.add_argument("--output_dir", default="./data/wikitext_processed", help="输出目录")
    parser.add_argument("--chunk_size", type=int, default=1000, help="每个二进制文件包含的样本数")
    parser.add_argument("--prediction_ratio", type=float, default=0.05,
                        help="尾追加token的比例，用于训练预测任务")
    parser.add_argument("--deletion_ratio", type=float, default=0.05,
                        help="插入token的比例，用于训练删除任务")
    parser.add_argument("--insertion_ratio", type=float, default=0.05,
                        help="删除token的比例，用于训练插入任务")
    parser.add_argument("--sample_rate", type=float, default=0.1,
                        help="从原始数据中采样的比例，减少生成的样本数量")
    args = parser.parse_args()

    # 创建输出目录
    prediction_dir = os.path.join(args.output_dir, "prediction")
    deletion_dir = os.path.join(args.output_dir, "deletion")
    insertion_dir = os.path.join(args.output_dir, "insertion")
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载tokenizer
    print(f"加载tokenizer: {args.base}")
    tokenizer = get_tokenizer(args.base, use_fast=True)

    # 词汇表大小，用于随机token生成
    vocab_size = tokenizer.vocab_size

    # 加载数据集
    print("加载WikiText-103数据集...")
    dataset = load_dataset("wikitext", "wikitext-103-v1")

    for split in ["train", "validation", "test"]:
        print(f"处理{split}集...")
        tokens_list = []

        # 处理数据集的每一行
        for item in tqdm(dataset[split]):
            text = item["text"]
            if not text.strip():  # 跳过空行
                continue

            # 对文本进行tokenize
            tokens = tokenizer.encode(text, add_special_tokens=False)
            tokens_list.extend(tokens)

        # 将token列表分成指定长度的序列
        num_sequences = len(tokens_list) // args.seq_len

        # 截断token列表使其能被序列长度整除
        tokens_list = tokens_list[:num_sequences * args.seq_len]

        # 重塑为序列的形状
        original_sequences = np.array(tokens_list).reshape(-1, args.seq_len)
        total_original_sequences = original_sequences.shape[0]

        print(f"总计生成了 {total_original_sequences} 个长度为 {args.seq_len} 的原始序列")

        # 初始化计数器和块索引
        prediction_chunk_idx = 0
        deletion_chunk_idx = 0
        insertion_chunk_idx = 0
        prediction_buffer = []
        deletion_buffer = []
        insertion_buffer = []
        total_prediction_samples = 0
        total_deletion_samples = 0
        total_insertion_samples = 0

        # 计算要采样的序列数量
        sample_size = int(total_original_sequences * args.sample_rate)
        if sample_size < 1000:  # 确保至少有1000个序列
            sample_size = min(1000, total_original_sequences)

        print(f"从中采样 {sample_size} 个序列进行编辑任务处理")

        # 随机采样序列
        sample_indices = np.random.choice(total_original_sequences, sample_size, replace=False)
        sampled_sequences = original_sequences[sample_indices]
        # ============================================================================
        # 为每个采样序列生成任务数据（修复版：每种编辑任务只产生 1 个缺陷样本）
        # ============================================================================
        for seq_idx in tqdm(range(len(sampled_sequences)), desc="创建任务数据"):
            # ---------- 共有预处理 ----------
            original_seq = sampled_sequences[seq_idx].tolist()  # list[int]
            seq_len = len(original_seq)
            if seq_len <= 1:  # 极短文本跳过
                continue

            # -----------------------------------------------------------------------
            # 1) 预测任务 (与旧实现一致，可一次生成多个样本)
            # -----------------------------------------------------------------------
            # 预测任务：生成预测位置样本
            num_to_predict = max(1, int(seq_len * args.prediction_ratio))
            predict_positions = sorted(random.sample(range(1, seq_len - 1), num_to_predict))

            for pos in predict_positions:
                next_token = original_seq[pos + 1]  # 目标 token
                pred_seq = original_seq[:pos + 1]
                # 如果长度不足，则使用pad_token_id进行补齐
                if len(pred_seq) < args.seq_len:
                    pred_seq = pred_seq + [tokenizer.pad_token_id] * (args.seq_len - len(pred_seq))
                # 如果长度超长（理论上不会发生，因为original_seq长度固定），也进行截断
                elif len(pred_seq) > args.seq_len:
                    pred_seq = pred_seq[:args.seq_len]
                prediction_buffer.append({
                    "sequence": torch.tensor(pred_seq),
                    "index": torch.tensor(pos + 1),
                    "token": torch.tensor(next_token)
                })
                total_prediction_samples += 1
                if len(prediction_buffer) >= args.chunk_size:
                    _save_prediction_chunk(
                        prediction_buffer, prediction_dir,
                        split, prediction_chunk_idx, tokenizer.pad_token_id
                    )
                    prediction_chunk_idx += 1
                    prediction_buffer = []

            # -----------------------------------------------------------------------
            # 2) 删除任务 —— 只插入 1 个随机 token
            # -----------------------------------------------------------------------
            # 删除任务：先在序列中插入随机 token
            num_to_delete = max(1, int(seq_len * args.deletion_ratio))
            to_delete_positions = sorted(random.sample(range(seq_len), num_to_delete))

            for to_delete_pos in to_delete_positions:
                del_seq = original_seq.copy()
                random_token = random.randint(0, vocab_size - 1)

                del_seq.insert(to_delete_pos, random_token)  # 执行插入
                # 如果插入后超过固定长度，则截断到args.seq_len
                if len(del_seq) > args.seq_len:
                    del_seq = del_seq[:args.seq_len]
                # 如果插入后长度不足，也进行补齐（一般情况不会发生）
                elif len(del_seq) < args.seq_len:
                    del_seq = del_seq + [tokenizer.pad_token_id] * (args.seq_len - len(del_seq))

                # 若截断导致缺陷 token 被裁掉，则跳过该样本
                if to_delete_pos < len(del_seq):
                    deletion_buffer.append({
                        "sequence": torch.tensor(del_seq),
                        "index": torch.tensor(to_delete_pos),
                        "token": torch.tensor(get_del_token_id(tokenizer))
                    })
                    total_deletion_samples += 1
                    if len(deletion_buffer) >= args.chunk_size:
                        _save_deletion_chunk(deletion_buffer, deletion_dir,
                                             split, deletion_chunk_idx
                                             )
                        deletion_chunk_idx += 1
                        deletion_buffer = []

            # -----------------------------------------------------------------------
            # 3) 插入任务 —— 只删除 1 个随机 token
            # -----------------------------------------------------------------------
            # 插入任务：先在序列中删除 1 个 token
            num_to_insert = max(1, int(seq_len * args.insertion_ratio))
            to_insert_positions = sorted(random.sample(range(seq_len), num_to_insert))
            for to_insert_pos in to_insert_positions:
                to_insert_token = original_seq[to_insert_pos]

                ins_seq = original_seq.copy()
                ins_seq.pop(to_insert_pos)  # 执行删除

                # 如果删除后长度不足，使用pad_token_id进行补齐到固定长度
                if len(ins_seq) < args.seq_len:
                    ins_seq = ins_seq + [tokenizer.pad_token_id] * (args.seq_len - len(ins_seq))
                # 如果意外超长，则截断（理论上不可能）
                elif len(ins_seq) > args.seq_len:
                    ins_seq = ins_seq[:args.seq_len]

                insertion_buffer.append({
                    "sequence": torch.tensor(ins_seq),
                    "index": torch.tensor(to_insert_pos),
                    "token": torch.tensor(to_insert_token)
                })
                total_insertion_samples += 1
                if len(insertion_buffer) >= args.chunk_size:
                    _save_insertion_chunk(insertion_buffer, insertion_dir,
                                          split, insertion_chunk_idx, tokenizer.pad_token_id)
                    insertion_chunk_idx += 1
                    insertion_buffer = []

        # 保存最后的缓冲区数据
        if prediction_buffer:
            _save_prediction_chunk(prediction_buffer, prediction_dir, split, prediction_chunk_idx,
                                   tokenizer.pad_token_id)
            prediction_chunk_idx += 1

        if deletion_buffer:
            _save_deletion_chunk(deletion_buffer, deletion_dir, split, deletion_chunk_idx)
            deletion_chunk_idx += 1

        if insertion_buffer:
            _save_insertion_chunk(insertion_buffer, insertion_dir, split, insertion_chunk_idx, tokenizer.pad_token_id)
            insertion_chunk_idx += 1

        print(f"生成了 {total_prediction_samples} 个预测任务样本，保存为 {prediction_chunk_idx} 个数据块")
        print(f"生成了 {total_deletion_samples} 个删除任务样本，保存为 {deletion_chunk_idx} 个数据块")
        print(f"生成了 {total_insertion_samples} 个插入任务样本，保存为 {insertion_chunk_idx} 个数据块")

        # 保存元数据
        metadata = {
            "num_prediction_chunks": prediction_chunk_idx,
            "num_deletion_chunks": deletion_chunk_idx,
            "num_insertion_chunks": insertion_chunk_idx,
            "chunk_size": args.chunk_size,
            "seq_len": args.seq_len,
            "vocab_size": vocab_size,
            "total_prediction_samples": total_prediction_samples,
            "total_deletion_samples": total_deletion_samples,
            "total_insertion_samples": total_insertion_samples
        }
        torch.save(metadata, os.path.join(args.output_dir, f"{split}_metadata.pt"))
        torch.save(metadata, os.path.join(prediction_dir, f"{split}_metadata.pt"))
        torch.save(metadata, os.path.join(deletion_dir, f"{split}_metadata.pt"))
        torch.save(metadata, os.path.join(insertion_dir, f"{split}_metadata.pt"))

    print("预处理完成！")


def _save_prediction_chunk(buffer, output_dir, split, chunk_idx, pad_token_id):  # 添加 pad_token_id 参数
    """辅助函数：保存预测任务数据块"""
    sequences = [item['sequence'] for item in buffer]
    indices = torch.stack([item['index'] for item in buffer])
    tokens = torch.stack([item['token'] for item in buffer])

    max_len = max(seq.size(0) for seq in sequences)
    # 使用 pad_token_id 初始化，而不是 0
    padded_sequences = torch.full((len(sequences), max_len), fill_value=pad_token_id, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded_sequences[i, :seq.size(0)] = seq

    data_chunk = {
        'sequences': padded_sequences,
        'indices': indices,
        'tokens': tokens
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{split}_prediction_chunk_{chunk_idx}.pt")
    torch.save(data_chunk, output_path)


def _save_deletion_chunk(buffer, output_dir, split, chunk_idx):
    """辅助函数：保存删除任务数据块"""
    # 将buffer中的数据转换为张量
    sequences = torch.stack([item['sequence'] for item in buffer])
    indices = torch.stack([item['index'] for item in buffer])
    tokens = torch.stack([item['token'] for item in buffer])

    # 保存数据块
    data_chunk = {
        'sequences': sequences,
        'indices': indices,
        'tokens': tokens
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{split}_deletion_chunk_{chunk_idx}.pt")
    torch.save(data_chunk, output_path)


def _save_insertion_chunk(buffer, output_dir, split, chunk_idx, pad_token_id):  # 添加 pad_token_id 参数
    """辅助函数：保存插入任务数据块"""
    # 将buffer中的数据转换为张量
    sequences = [item['sequence'] for item in buffer]
    indices = torch.stack([item['index'] for item in buffer])
    tokens = torch.stack([item['token'] for item in buffer])

    max_len = max(seq.size(0) for seq in sequences)
    # 使用 pad_token_id 初始化，而不是 0
    padded_sequences = torch.full((len(sequences), max_len), fill_value=pad_token_id, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded_sequences[i, :seq.size(0)] = seq

    # 保存数据块
    data_chunk = {
        'sequences': padded_sequences,
        'indices': indices,
        'tokens': tokens
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{split}_insertion_chunk_{chunk_idx}.pt")
    torch.save(data_chunk, output_path)


if __name__ == "__main__":
    main()
