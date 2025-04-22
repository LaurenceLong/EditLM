from typing import Optional, Dict

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


# ----------------------------------------------
class EditLMHF(nn.Module):
    """
    复用 HuggingFace 预训练模型作为 backbone，仅新增：
      1. triple_proj  : 3D -> D
      2. index_head   : gap 位置分类
      3. token_head   : 插入 token 预测（共享词嵌入）
    """

    def __init__(self,
                 base_model: str = "facebook/opt-125m",
                 index_loss_weight: float = 1.0):
        super().__init__()

        self.backbone = AutoModelForCausalLM.from_pretrained(
            base_model,
            # torch_dtype=torch.float16,  # 直接以 fp16 载入，省内存+显存
            low_cpu_mem_usage=True
        )
        self.backbone.requires_grad_(True)  # 若想冻结，只需改成 False

        # ---- 一些通用属性 ----------------------------------------------------
        config = self.backbone.config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.index_loss_weight = index_loss_weight

        # ---- EditLM 额外模块 --------------------------------------------------
        self.triple_proj = nn.Linear(3 * self.hidden_size, self.hidden_size, bias=False)
        self.index_head = nn.Linear(self.hidden_size, 1, bias=False)
        self.token_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        # 词向量权重共享 - 根据不同模型架构可能需要调整
        try:
            self.token_head.weight = self.backbone.get_input_embeddings().weight
        except AttributeError:
            # 防止某些模型没有标准的get_input_embeddings方法
            if hasattr(self.backbone, 'transformer') and hasattr(self.backbone.transformer, 'wte'):
                # GPT类模型
                self.token_head.weight = self.backbone.transformer.wte.weight
            elif hasattr(self.backbone, 'model') and hasattr(self.backbone.model, 'embed_tokens'):
                # OPT类模型
                self.token_head.weight = self.backbone.model.embed_tokens.weight
            else:
                print("Warning: Unable to share token embedding weights automatically")

    # -------------------------------------------------------------------------
    def _build_gap_state(self, h: torch.Tensor) -> torch.Tensor:
        """
        h : [B, L, D]  ->  gap_state : [B, L+1, D]
        """
        B, L, D = h.size()
        z = torch.zeros(B, 1, D, device=h.device, dtype=h.dtype)

        # 修复: 确保所有张量都有相同的长度
        h_l = torch.cat([z, h[:, :-1]], 1)  # 左上下文 [B, L, D]
        h_c = h  # 中间上下文 [B, L, D]
        h_r = torch.cat([h[:, 1:], z], 1)  # 右上下文 [B, L, D]

        # 为了创建L+1长度的结果，我们需要额外处理边界情况
        # 添加最后一个gap位置（句子结束后）
        h_l_last = h[:, -1:, :]  # 最后一个token作为左上下文
        h_c_last = z  # 句子结束没有中间上下文
        h_r_last = z  # 句子结束没有右上下文

        # 拼接所有位置信息
        h_l = torch.cat([h_l, h_l_last], 1)  # [B, L+1, D]
        h_c = torch.cat([h_c, h_c_last], 1)  # [B, L+1, D]
        h_r = torch.cat([h_r, h_r_last], 1)  # [B, L+1, D]

        # 组合三元组表示
        triple = torch.cat([h_l, h_c, h_r], dim=-1)  # [B, L+1, 3D]
        return self.triple_proj(triple)  # [B, L+1, D]

    # -------------------------------------------------------------------------
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                target_index: Optional[torch.Tensor] = None,
                target_token: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        """
        训练模式：传入 target_index & target_token，返回带 loss 的 dict
        推理模式：仅传 input_ids，返回 logits
        """
        # HF backbone 返回 last_hidden_state [B, L, D]
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # 获取最后一层隐藏状态
        if hasattr(outputs, 'last_hidden_state'):
            h = outputs.last_hidden_state
        else:
            h = outputs.hidden_states[-1]

        gap = self._build_gap_state(h)  # [B, L+1, D]
        idx_logits = self.index_head(gap).squeeze(-1)  # [B, L+1]
        tok_logits_all = self.token_head(gap)  # [B, L+1, V]

        if target_index is None:  # ------ 推理 --------------------
            pred_index = idx_logits.argmax(-1)  # [B]
            gather = pred_index.view(-1, 1, 1).expand(-1, 1, tok_logits_all.size(-1))
            tok_logits = tok_logits_all.gather(1, gather).squeeze(1)
            return dict(index_logits=idx_logits, token_logits=tok_logits)

        # --------------------------- 训练 -------------------------------------
        B = input_ids.size(0)
        gather = target_index.view(B, 1, 1).expand(-1, 1, tok_logits_all.size(-1))
        tok_logits = tok_logits_all.gather(1, gather).squeeze(1)  # [B, V]

        tok_loss = nn.functional.cross_entropy(tok_logits, target_token)
        idx_loss = nn.functional.cross_entropy(idx_logits, target_index)
        loss = tok_loss + self.index_loss_weight * idx_loss

        return dict(loss=loss,
                    tok_loss=tok_loss,
                    idx_loss=idx_loss,
                    index_logits=idx_logits,
                    token_logits=tok_logits)
