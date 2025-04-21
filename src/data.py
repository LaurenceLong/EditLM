import os

import torch
import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast

from config import TrainConfig

WP = os.path.expanduser("~/.cache/editlm")

tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained('Xenova/gpt-4o')


class WikiDataset(Dataset):
    def __init__(self, split: str, cfg: TrainConfig):
        self.ids = []
        ds = load_dataset("wikitext", "wikitext-103-v1", split=split)
        for line in tqdm.tqdm(ds["text"], desc="tokenizing"):
            ids = tokenizer.encode(line, return_tensors="pt")
            if len(ids) == 0:
                continue
            # chunk
            for i in range(0, len(ids) - cfg.seq_len, cfg.seq_len):
                self.ids.append(torch.tensor(ids[i:i + cfg.seq_len]))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i]


def get_loader(split: str, cfg: TrainConfig):
    ds = WikiDataset(split, cfg)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
