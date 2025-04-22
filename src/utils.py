import glob
import math
import os

import torch
from transformers import AutoTokenizer

from model_hf import EditLMHF
from tokenizer import get_tokenizer


def save_ckpt(model, opt, step, path, tokenizer=None):
    """保存检查点，包括模型、优化器和tokenizer"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'opt': opt.state_dict(),
        'step': step
    }, path)
    print(f"Checkpoint saved to {path}")

    # 如果提供了tokenizer，也保存它
    if tokenizer is not None:
        # 从路径中提取目录
        directory = os.path.dirname(path)
        # 保存tokenizer到同一个目录
        tokenizer.save_pretrained(directory)
        print(f"Tokenizer saved to {directory}")


def load_model_from_ckpt(ckpt_path, base_model=None):
    """从检查点加载模型和tokenizer"""
    # 加载检查点
    ckpt = torch.load(ckpt_path, map_location='cpu')

    # 从检查点路径获取目录
    directory = os.path.dirname(ckpt_path)

    # 确定是否存在tokenizer文件
    tokenizer_files = glob.glob(os.path.join(directory, "tokenizer_config.json"))

    if tokenizer_files:
        # 从同一目录加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(directory)
        print(f"Loaded tokenizer from {directory}")
    else:
        # 回退到base_model
        if base_model:
            tokenizer = get_tokenizer(base_model, use_fast=True)
            print(f"Loaded tokenizer from base model: {base_model}")
        else:
            raise ValueError("No tokenizer found and no base_model provided")

    # 创建模型
    model = EditLMHF(base_model=base_model if base_model else directory)
    model.load_state_dict(ckpt['model'])

    return model, tokenizer, ckpt.get('step', 0)


class WarmupCosine:
    def __init__(self, opt, cfg, total_steps):
        self.opt, self.cfg, self.total = opt, cfg, total_steps
        self.step_ = 0

    def step(self):
        self.step_ += 1
        lr_ratio = min(self.step_ / self.cfg.warmup_steps,
                       0.5 * (1 + math.cos(math.pi * (self.step_ - self.cfg.warmup_steps) /
                                           (self.total - self.cfg.warmup_steps))))
        for g in self.opt.param_groups:
            g["lr"] = self.cfg.lr * lr_ratio

