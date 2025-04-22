import math
import torch
import torch.nn as nn
from typing import Optional, Dict

from torch.nn import functional as F

from config import ModelConfig


# -------------------- Rotary -------------------------------------------------
def rope(x, seq_len, theta=10000.0):
    """Applies RoPE to q/k: x [B, H, L, D]"""
    dim = x.size(-1)
    assert dim % 2 == 0
    freqs = torch.arange(0, dim, 2, device=x.device) / dim
    freqs = 1.0 / (theta ** freqs)
    t = torch.arange(seq_len, device=x.device)
    freqs = torch.outer(t, freqs)  # [L, D/2]
    sin, cos = freqs.sin(), freqs.cos()  # [L, D/2]

    sin = torch.repeat_interleave(sin, 2, -1)  # [L, D]
    cos = torch.repeat_interleave(cos, 2, -1)
    x1, x2 = x[..., ::2], x[..., 1::2]
    x_rot = torch.stack([-x2, x1], -1).reshape_as(x)
    return x * cos + x_rot * sin


# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.heads = cfg.n_heads
        self.dim = cfg.dim
        self.scale = 1 / math.sqrt(cfg.dim // cfg.n_heads)
        self.qkv = nn.Linear(cfg.dim, cfg.dim * 3, bias=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x, mask=None):
        B, L, D = x.size()
        qkv = self.qkv(x).view(B, L, 3, self.heads, D // self.heads)
        q, k, v = qkv.unbind(2)  # each [B, L, H, D/H]
        q = rope(q, L)
        k = rope(k, L)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]  # [B, H, L, D/H]
        att = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            att = att.masked_fill(mask == 0, -1e9)
        att = att.softmax(-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.dim, 4 * cfg.dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * cfg.dim, cfg.dim, bias=False)
        )

    def forward(self, x): return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn = Attention(cfg)
        self.ff = FeedForward(cfg)
        self.ln1 = nn.LayerNorm(cfg.dim)
        self.ln2 = nn.LayerNorm(cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, mask=None):
        x = x + self.drop(self.attn(self.ln1(x), mask))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


# --------------------- EditLM ------------------------------------------------
class EditLM(nn.Module):
    def __init__(self, cfg: ModelConfig, pad_id: int, eos_id: int):
        super().__init__()
        self.cfg = cfg
        self.pad_id, self.eos_id = pad_id, eos_id

        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_ln = nn.LayerNorm(cfg.dim)

        # ----------- 三元组 gap 表征 -------------
        self.triple_proj = nn.Linear(3 * cfg.dim, cfg.dim, bias=False)
        self.index_head = nn.Linear(cfg.dim, 1, bias=False)
        self.token_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        # 权重共享
        self.token_head.weight = self.embed.weight

    # ---------------------------------------------------------------------
    def _make_mask(self, seq_len, device):
        return torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

    def _build_gap_state(self, h):
        B, L, D = h.size()
        z = torch.zeros(B, 1, D, device=h.device, dtype=h.dtype)
        h_l = torch.cat([z, h[:, :-1]], 1)
        h_c = torch.cat([h, z], 1)
        h_r = torch.cat([h[:, 1:], z], 1)
        triple = torch.cat([h_l, h_c, h_r], -1)  # [B,L+1,3D]
        return self.triple_proj(triple)  # [B,L+1,D]

    # ---------------------------------------------------------------------
    def forward(self,
                input_ids: torch.Tensor,
                target_index: Optional[torch.Tensor] = None,
                target_token: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        B, L = input_ids.shape
        mask = self._make_mask(L, input_ids.device)

        x = self.embed(input_ids)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x, mask)
        h = self.final_ln(x)  # [B,L,D]

        gap = self._build_gap_state(h)  # [B,L+1,D]
        idx_logits = self.index_head(gap).squeeze(-1)  # [B,L+1]
        tok_logits_all = self.token_head(gap)  # [B,L+1,V]

        if target_index is not None:  # 训练
            gather = target_index.view(B, 1, 1).expand(-1, 1, tok_logits_all.size(-1))
            tok_logits = tok_logits_all.gather(1, gather).squeeze(1)  # [B,V]
        else:  # 推理
            pred_index = idx_logits.argmax(-1)
            gather = pred_index.view(B, 1, 1).expand(-1, 1, tok_logits_all.size(-1))
            tok_logits = tok_logits_all.gather(1, gather).squeeze(1)

        out = dict(index_logits=idx_logits, token_logits=tok_logits)
        if target_index is not None and target_token is not None:
            tok_loss = F.cross_entropy(tok_logits, target_token)
            idx_loss = F.cross_entropy(idx_logits, target_index)
            loss = tok_loss + self.cfg.index_loss_weight * idx_loss
            out.update(loss=loss, tok_loss=tok_loss, idx_loss=idx_loss)
        return out

