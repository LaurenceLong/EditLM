import argparse
import torch
import tqdm
from torch.cuda.amp import autocast, GradScaler

from config import ModelConfig, TrainConfig
from data import get_loader, tokenizer
from model import EditLM
from utils import save_ckpt, WarmupCosine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    mcfg = ModelConfig(vocab_size=tokenizer.vocab_size)
    tcfg = TrainConfig()
    device = "cuda"

    model = EditLM(mcfg, pad_id=tokenizer.pad_token_id, eos_id=tokenizer.eos_token_id).to(device)
    if tcfg.fp16:
        model = model.half()
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, betas=(0.9, 0.95), weight_decay=0.1)
    sched = WarmupCosine(opt, tcfg, tcfg.total_steps)

    scaler = GradScaler(enabled=tcfg.fp16)
    dl = get_loader("train", tcfg)
    it = iter(dl)

    pbar = tqdm.trange(1, tcfg.total_steps + 1)
    for step in pbar:
        try:
            x = next(it)
        except StopIteration:
            it = iter(dl)
            x = next(it)
        x = x.to(device)
        # teacher signal：将序列最后一个 gap 设为 gold
        target_index = torch.full((tcfg.batch_size,), fill_value=tcfg.seq_len, device=device)
        target_token = x[:, -1]  # 预测下一个 token (GPT‑2 shift trick)

        with autocast(enabled=tcfg.fp16):
            out = model(x, target_index, target_token)
            loss = out["loss"] / tcfg.batch_size
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        sched.step()

        pbar.set_description(f"loss {loss.item():.3f}")
        if step % tcfg.ckpt_every == 0:
            save_ckpt(model, opt, step, f"{args.outdir}/step{step}.pt")


if __name__ == "__main__":
    main()
