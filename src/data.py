# src/data.py (or wherever your datasets are defined)
import os
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

WP = os.path.expanduser("~/.cache/editlm")


# 删除任务数据集
# 修改DeletionTaskDataset类的初始化顺序
class DeletionTaskDataset(Dataset):
    def __init__(self, data_dir, split, shuffle=True, del_token_id=None):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.shuffle = shuffle
        self.del_token_id = del_token_id

        # 加载元数据
        metadata_path = os.path.join(data_dir, f"{split}_metadata.pt")
        self.metadata = torch.load(metadata_path)

        self.num_chunks = self.metadata["num_deletion_chunks"]
        self.chunk_size = self.metadata["chunk_size"]
        self.seq_len = self.metadata["seq_len"]

        # 创建所有块的索引顺序（用于随机访问）- 修改顺序以避免错误
        if self.shuffle:
            self.chunk_order = list(range(self.num_chunks))
            random.shuffle(self.chunk_order)
        else:
            self.chunk_order = list(range(self.num_chunks))

        # 加载第一个数据块
        self.current_chunk_idx = 0
        self.current_data = self._load_chunk(0)

    def _load_chunk(self, chunk_idx):
        """加载指定索引的数据块"""
        actual_chunk_idx = self.chunk_order[chunk_idx % self.num_chunks]
        path = os.path.join(self.data_dir, f"{self.split}_deletion_chunk_{actual_chunk_idx}.pt")
        return torch.load(path)

    def __len__(self):
        # 粗略估计总长度
        return self.num_chunks * self.chunk_size

    def __getitem__(self, idx):
        # 计算数据在哪个块
        chunk_idx = idx // self.chunk_size
        local_idx = idx % self.chunk_size

        # 如果需要，加载新的数据块
        if chunk_idx != self.current_chunk_idx:
            self.current_chunk_idx = chunk_idx
            self.current_data = self._load_chunk(chunk_idx)

        # 如果索引超出当前块的范围，则回滚到第一个块
        if local_idx >= len(self.current_data['sequences']):
            self.current_chunk_idx = 0
            self.current_data = self._load_chunk(0)
            local_idx = idx % len(self.current_data['sequences'])

        return {
            'sequence': self.current_data['sequences'][local_idx],
            'index': self.current_data['indices'][local_idx],
            'token': torch.tensor(self.del_token_id)  # 删除任务的token标记为
        }


# 插入任务数据集
# 修改InsertionTaskDataset类的初始化顺序
class InsertionTaskDataset(Dataset):
    def __init__(self, data_dir, split, shuffle=True):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.shuffle = shuffle

        # 加载元数据
        metadata_path = os.path.join(data_dir, f"{split}_metadata.pt")
        self.metadata = torch.load(metadata_path)

        self.num_chunks = self.metadata["num_insertion_chunks"]
        self.chunk_size = self.metadata["chunk_size"]
        self.seq_len = self.metadata["seq_len"]

        # 创建所有块的索引顺序（用于随机访问）- 修改顺序以避免错误
        if self.shuffle:
            self.chunk_order = list(range(self.num_chunks))
            random.shuffle(self.chunk_order)
        else:
            self.chunk_order = list(range(self.num_chunks))

        # 加载第一个数据块
        self.current_chunk_idx = 0
        self.current_data = self._load_chunk(0)

    def _load_chunk(self, chunk_idx):
        """加载指定索引的数据块"""
        actual_chunk_idx = self.chunk_order[chunk_idx % self.num_chunks]
        path = os.path.join(self.data_dir, f"{self.split}_insertion_chunk_{actual_chunk_idx}.pt")
        return torch.load(path)

    def __len__(self):
        # 粗略估计总长度
        return self.num_chunks * self.chunk_size

    def __getitem__(self, idx):
        # 计算数据在哪个块
        chunk_idx = idx // self.chunk_size
        local_idx = idx % self.chunk_size

        # 如果需要，加载新的数据块
        if chunk_idx != self.current_chunk_idx:
            self.current_chunk_idx = chunk_idx
            self.current_data = self._load_chunk(chunk_idx)

        # 如果索引超出当前块的范围，则回滚到第一个块
        if local_idx >= len(self.current_data['sequences']):
            self.current_chunk_idx = 0
            self.current_data = self._load_chunk(0)
            local_idx = idx % len(self.current_data['sequences'])

        return {
            'sequence': self.current_data['sequences'][local_idx],
            'index': self.current_data['indices'][local_idx],
            'token': self.current_data['tokens'][local_idx]
        }


class PredictionTaskDataset(Dataset):
    def __init__(self, data_dir, split, shuffle=True):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.shuffle = shuffle

        # 加载元数据
        metadata_path = os.path.join(data_dir, f"{split}_metadata.pt")
        self.metadata = torch.load(metadata_path)

        self.num_chunks = self.metadata["num_prediction_chunks"]
        self.chunk_size = self.metadata["chunk_size"]
        self.seq_len = self.metadata["seq_len"]

        # 创建所有块的索引顺序（用于随机访问）- 修改顺序以避免错误
        if self.shuffle:
            self.chunk_order = list(range(self.num_chunks))
            random.shuffle(self.chunk_order)
        else:
            self.chunk_order = list(range(self.num_chunks))

        # 加载第一个数据块
        self.current_chunk_idx = 0
        self.current_data = self._load_chunk(0)

    def _load_chunk(self, chunk_idx):
        """加载指定索引的数据块"""
        actual_chunk_idx = self.chunk_order[chunk_idx % self.num_chunks]
        path = os.path.join(self.data_dir, f"{self.split}_prediction_chunk_{actual_chunk_idx}.pt")
        return torch.load(path)

    def __len__(self):
        # 粗略估计总长度
        return self.num_chunks * self.chunk_size

    def __getitem__(self, idx):
        # 计算数据在哪个块
        chunk_idx = idx // self.chunk_size
        local_idx = idx % self.chunk_size

        # 如果需要，加载新的数据块
        if chunk_idx != self.current_chunk_idx:
            self.current_chunk_idx = chunk_idx
            self.current_data = self._load_chunk(chunk_idx)

        # 如果索引超出当前块的范围，则回滚到第一个块
        if local_idx >= len(self.current_data['sequences']):
            self.current_chunk_idx = 0
            self.current_data = self._load_chunk(0)
            local_idx = idx % len(self.current_data['sequences'])

        return {
            'sequence': self.current_data['sequences'][local_idx],
            'index': self.current_data['indices'][local_idx],
            'token': self.current_data['tokens'][local_idx]
        }


def create_collate_fn(pad_token_id: int):
    """
    创建一个 collate 函数，填充序列并生成 attention mask
    """

    def collate_fn(batch):
        # 取出所有 sequence（可能长度不同）
        sequences = [item['sequence'] for item in batch]
        # 使用 pad_sequence 将其填充成同一长度
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=pad_token_id)

        # indices 和 tokens 假设单个标量或已经统一形状，可以直接堆叠
        indices = torch.stack([item['index'] for item in batch])
        tokens = torch.stack([item['token'] for item in batch])

        # 生成 attention mask：非 pad_token 的位置标记为 1
        attention_mask = (padded_sequences != pad_token_id).long()

        return {
            'sequences': padded_sequences,
            'attention_mask': attention_mask,
            'indices': indices,
            'tokens': tokens
        }

    return collate_fn
