import math
import os
import torch


def save_ckpt(model, opt, step, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"step": step,
                "model": model.state_dict(),
                "opt": opt.state_dict()}, path)


def load_ckpt(model, opt, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    return ckpt["step"]


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
