import argparse
import torch
import tqdm

from config import ModelConfig, TrainConfig
from data import get_loader, tokenizer
from model import EditLM
from utils import load_ckpt, save_ckpt


def compute_reward(logprobs):  # 简单示例：logp(up) 越高奖励越大
    return -logprobs.mean(dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    device = "cuda"
    mcfg = ModelConfig(vocab_size=tokenizer.eos_token_id)
    tcfg = TrainConfig(batch_size=8, total_steps=20_000, lr=5e-5)

    policy = EditLM(mcfg, tokenizer.pad_token_id, tokenizer.eos_token_id).to(device).half()
    ref = EditLM(mcfg, tokenizer.pad_token_id, tokenizer.eos_token_id).to(device).half().eval()

    opt = torch.optim.AdamW(policy.parameters(), lr=tcfg.lr)
    step0 = load_ckpt(policy, opt, args.ckpt, device)

    dl = get_loader("validation", tcfg)
    iterator = iter(dl)

    for step in tqdm.trange(step0, tcfg.total_steps + step0):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dl)
            batch = next(iterator)
        batch = batch.to(device)

        # rollout 1 步：预测 插入末尾 + token
        with torch.no_grad():
            out = policy(batch)
            idx = out["index_logits"].argmax(-1)
            tok = torch.distributions.Categorical(logits=out["token_logits"]).sample()
        # log‑p
        logp_old = torch.distributions.Categorical(logits=out["token_logits"]).log_prob(tok)

        # 计算 reward
        reward = compute_reward(logp_old.unsqueeze(1))

        # PPO clip update
        for _ in range(4):
            pol_out = policy(batch, target_index=idx, target_token=tok)
            logp = torch.distributions.Categorical(logits=pol_out["token_logits"]).log_prob(tok)
            ratio = torch.exp(logp - logp_old)
            adv = reward - reward.mean()
            clip = 0.2
            loss = -(torch.minimum(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv)).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

        if step % 1000 == 0:
            save_ckpt(policy, opt, step, f"{args.outdir}/rl_step{step}.pt")


if __name__ == "__main__":
    main()

