"""
预训练数据生成和处理模块
"""
import random
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple
import numpy as np

class TextShiftDataset(Dataset):
    """
    用于预训练的文本shift数据集
    通过随机变换生成源文本和目标文本对
    """
    def __init__(self, corpus_path, tokenizer, config, max_samples=None):
        self.tokenizer = tokenizer
        self.config = config
        self.max_seq_length = config.max_seq_length
        self.samples = self.prepare_data(corpus_path, max_samples)

    def prepare_data(self, corpus_path, max_samples):
        """
        从语料库准备预训练数据
        生成带有多种编辑操作的文本对
        """
        samples = []

        # 读取语料库
        with open(corpus_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 处理每个文本块
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # 词元化文本
            tokens = self.tokenizer.encode(line, add_special_tokens=False)

            # 如果文本太短，跳过
            if len(tokens) < 10:
                continue

            # 生成多个编辑样本
            for _ in range(min(3, len(tokens) // 10)):  # 每个文本生成多个样本
                # 裁剪到合适长度
                if len(tokens) > self.max_seq_length - 10:  # 留出一些空间用于编辑
                    start_idx = random.randint(0, len(tokens) - (self.max_seq_length - 10))
                    text_segment = tokens[start_idx:start_idx + self.max_seq_length - 10]
                else:
                    text_segment = tokens.copy()

                # 生成编辑版本
                source_text, target_text = self.generate_edit_pair(text_segment)

                samples.append({
                    "source": source_text,
                    "target": target_text
                })

                if max_samples and len(samples) >= max_samples:
                    return samples

        return samples

    def generate_edit_pair(self, text):
        """
        通过随机应用编辑操作生成源文本和目标文本对
        """
        source = text.copy()
        target = text.copy()

        # 随机决定要执行的编辑数量 (1-5)
        num_edits = random.randint(1, min(5, len(text) // 5))

        for _ in range(num_edits):
            edit_type = random.choice(['insert', 'delete', 'replace'])

            if edit_type == 'insert':
                # 随机插入1-3个词元
                insert_pos = random.randint(0, len(target))
                num_tokens = random.randint(1, 3)

                # 从词汇表中随机选择词元或从当前文本中采样
                if random.random() < 0.5:
                    # 从当前文本中随机采样词元
                    sampled_positions = random.sample(range(len(text)), min(num_tokens, len(text)))
                    insert_tokens = [text[pos] for pos in sampled_positions]
                else:
                    # 从词汇表中随机选择词元
                    vocab_size = self.tokenizer.vocab_size
                    insert_tokens = [random.randint(0, vocab_size-1) for _ in range(num_tokens)]

                # 执行插入
                for i, token in enumerate(insert_tokens):
                    target.insert(insert_pos + i, token)

            elif edit_type == 'delete' and len(target) > 5:
                # 随机删除1-3个连续词元
                if len(target) <= 3:
                    continue

                del_start = random.randint(0, len(target) - 1)
                del_len = random.randint(1, min(3, len(target) - del_start))

                # 执行删除
                for _ in range(del_len):
                    if del_start < len(target):
                        target.pop(del_start)

            elif edit_type == 'replace' and len(target) > 0:
                # 随机替换1-3个连续词元
                replace_start = random.randint(0, len(target) - 1)
                replace_len = random.randint(1, min(3, len(target) - replace_start))

                # 生成替换词元
                vocab_size = self.tokenizer.vocab_size
                replace_tokens = [random.randint(0, vocab_size-1) for _ in range(replace_len)]

                # 执行替换
                for i in range(replace_len):
                    if replace_start + i < len(target):
                        target[replace_start + i] = replace_tokens[i]

        return source, target

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """返回源文本-目标文本对"""
        return self.samples[idx]

class PretrainingDataset(Dataset):
    """预训练数据集，处理文本shift生成的源文本-目标文本对"""

    def __init__(self, samples, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.samples = self.process_samples(samples)

    def process_samples(self, samples):
        """处理样本，生成训练数据"""
        from data.preprocess import generate_training_samples

        processed_data = []
        for sample in samples:
            source = sample["source"]
            target = sample["target"]

            # 生成训练样本
            training_samples = generate_training_samples(source, target, self.tokenizer)
            processed_data.extend(training_samples)

        return processed_data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """获取训练样本"""
        sample = self.samples[idx]

        # 准备输入序列
        input_ids = sample["input_state"]

        # 截断到最大长度
        if len(input_ids) > self.config.max_seq_length - 1:
            input_ids = input_ids[:self.config.max_seq_length - 1]

        # 创建注意力掩码
        attention_mask = [1] * len(input_ids)

        # 准备标签
        target_token_id = sample["target_token_id"]
        target_index = min(sample["target_index"], len(input_ids))

        # 添加语言模型目标（下一个词预测）用于多任务学习
        # 随机选择一个位置进行掩码
        if len(input_ids) > 1:
            mask_pos = random.randint(0, len(input_ids) - 1)
            lm_target = input_ids[mask_pos]
            lm_position = mask_pos
        else:
            lm_target = 0
            lm_position = 0

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "target_token_id": torch.tensor(target_token_id, dtype=torch.long),
            "target_index": torch.tensor(target_index, dtype=torch.long),
            "lm_target": torch.tensor(lm_target, dtype=torch.long),
            "lm_position": torch.tensor(lm_position, dtype=torch.long)
        }

def pretrain_collate_fn(batch, tokenizer):
    """预训练批处理函数"""
    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids_batch = []
    attention_mask_batch = []
    target_token_id_batch = []
    target_index_batch = []
    lm_target_batch = []
    lm_position_batch = []

    for item in batch:
        input_ids = item["input_ids"]
        attention_mask = item["attention_mask"]

        # 计算填充长度
        pad_len = max_len - len(input_ids)

        # 填充输入ID和注意力掩码
        input_ids = torch.cat([input_ids, torch.tensor([tokenizer.pad_token_id] * pad_len, dtype=torch.long)])
        attention_mask = torch.cat([attention_mask, torch.tensor([0] * pad_len, dtype=torch.long)])

        input_ids_batch.append(input_ids)
        attention_mask_batch.append(attention_mask)
        target_token_id_batch.append(item["target_token_id"])
        target_index_batch.append(item["target_index"])
        lm_target_batch.append(item["lm_target"])
        lm_position_batch.append(item["lm_position"])

    return {
        "input_ids": torch.stack(input_ids_batch),
        "attention_mask": torch.stack(attention_mask_batch),
        "target_token_ids": torch.stack(target_token_id_batch),
        "target_index_positions": torch.stack(target_index_batch),
        "lm_targets": torch.stack(lm_target_batch),
        "lm_positions": torch.stack(lm_position_batch)
    }
