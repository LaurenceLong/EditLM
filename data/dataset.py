"""
数据集类，用于加载和处理数据
"""
import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any
from data.preprocess import generate_training_samples

class EditDataset(Dataset):
    """编辑模型的数据集"""

    def __init__(self, file_path: str, tokenizer, config, is_training: bool = True):
        self.tokenizer = tokenizer
        self.config = config
        self.is_training = is_training
        self.samples = self.load_data(file_path)

    def load_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        加载数据文件并生成训练样本

        期望的数据格式：
        [
            {
                "source": "原始文本",
                "target": "目标文本"
            },
            ...
        ]
        """
        processed_samples = []

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            source = item["source"]
            target = item["target"]

            # 对源文本和目标文本进行词元化
            source_ids = self.tokenizer.encode(source, add_special_tokens=False)
            target_ids = self.tokenizer.encode(target, add_special_tokens=False)

            # 生成训练样本
            training_samples = generate_training_samples(source_ids, target_ids, self.tokenizer)
            processed_samples.extend(training_samples)

        return processed_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 准备输入序列
        input_ids = sample["input_state"]

        # 截断到最大长度(减1以保留位置给最后一个特殊标记)
        if len(input_ids) > self.config.max_seq_length - 1:
            input_ids = input_ids[:self.config.max_seq_length - 1]

        # 创建注意力掩码
        attention_mask = [1] * len(input_ids)

        # 准备标签
        target_token_id = sample["target_token_id"]
        target_index = min(sample["target_index"], len(input_ids))  # 确保索引不超过序列长度

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "target_token_id": torch.tensor(target_token_id, dtype=torch.long),
            "target_index": torch.tensor(target_index, dtype=torch.long)
        }

def collate_fn(batch, tokenizer):
    """
    数据批处理函数，将批次数据填充到相同长度
    """
    # 获取批次中最大序列长度
    max_len = max(len(item["input_ids"]) for item in batch)

    # 填充批次数据
    input_ids_batch = []
    attention_mask_batch = []
    target_token_id_batch = []
    target_index_batch = []

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

    return {
        "input_ids": torch.stack(input_ids_batch),
        "attention_mask": torch.stack(attention_mask_batch),
        "target_token_ids": torch.stack(target_token_id_batch),
        "target_index_positions": torch.stack(target_index_batch)
    }
