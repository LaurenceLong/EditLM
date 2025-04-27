from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


# --- 新增：轻量级 Transformer Encoder ---
class GapEncoder(nn.Module):
    """
    一个轻量级的 Transformer Encoder，用于处理插入了 GAP 标记的序列。
    它接收 backbone 的隐藏状态，通过降维、多层自注意力处理，再升维，
    输出增强后的表示，特别是 GAP 位置的表示。
    """

    def __init__(self,
                 in_dim: int,
                 hidden_dim: int = 1024,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 norm_first: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        # 降维投影层
        self.proj_down = nn.Linear(in_dim, hidden_dim, bias=False)

        # 标准 Transformer Encoder 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,  # 标准设置
            dropout=dropout,
            activation=activation,
            batch_first=True,  # 输入/输出格式为 [B, L, D]
            norm_first=norm_first  # Pre-LN 结构，通常更稳定
        )
        # Transformer Encoder 堆叠
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim) if norm_first else None  # Post-LN (如果 norm_first=False)
        )

        # 升维投影层 (可选，如果希望输出维度与输入一致)
        self.proj_up = nn.Linear(hidden_dim, in_dim, bias=False)

        print(
            f"GapEncoder initialized: in_dim={in_dim}, hidden_dim={hidden_dim}, layers={num_layers}, heads={num_heads}")
        param_count = sum(p.numel() for p in self.parameters())
        print(f"GapEncoder parameter count: {param_count / 1e6:.2f}M")

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: 输入序列 [B, L_interleaved, D_in]
            src_key_padding_mask: 可选的 padding mask [B, L_interleaved] (True 表示 padding)

        Returns:
            编码后的序列 [B, L_interleaved, D_in]
        """
        # 1. 降维
        hidden = self.proj_down(x)  # [B, L_interleaved, d_hidden]

        # 2. 通过 Transformer Encoder 处理
        # 注意：TransformerEncoder 需要 mask 指示哪些是 padding (True)
        encoded = self.encoder(hidden, src_key_padding_mask=src_key_padding_mask)  # [B, L_interleaved, d_hidden]

        # 3. 升维
        output = self.proj_up(encoded)  # [B, L_interleaved, D_in]

        return output


# --- 修改后的 EditLMHF ---
class EditLMHF(nn.Module):
    """
    改进版 EditLMHF，使用独立的 GapEncoder 处理上下文，替代原有的手动融合策略。

    架构:
    1. Backbone (e.g., Qwen) 提取最后一层 hidden_states。
    2. 将 hidden_states 序列插入 GAP 占位符，形成 [gap0, tok0, gap1, tok1, ..., gapL] 的交错序列。
    3. 将交错序列输入轻量级的 GapEncoder (Transformer Encoder)。
    4. 从 GapEncoder 的输出中提取 GAP 位置的表示，作为最终的 gap_state。
    5. 使用 index_head 和 edit_head 在 gap_state 上进行预测。
    """

    def __init__(self,
                 base_model: str = "Qwen/Qwen2.5-0.5B",
                 index_loss_weight: float = 1.0,
                 gap_encoder_layers: int = 2,
                 gap_encoder_heads: int = 8):
        super().__init__()

        print(f"Initializing EditLMHF_New with base model: {base_model}")
        self.backbone = AutoModelForCausalLM.from_pretrained(
            base_model,
            low_cpu_mem_usage=True,
        )

        self.backbone.requires_grad_(True)

        # Basic attributes
        config = self.backbone.config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size  # Backbone 的隐藏层维度 (D)
        self.index_loss_weight = index_loss_weight

        # --- 新增 GapEncoder ---
        self.gap_encoder = GapEncoder(
            in_dim=self.hidden_size,
            hidden_dim=self.hidden_size // 4,
            num_layers=gap_encoder_layers,
            num_heads=gap_encoder_heads
        )

        # --- 输出头 (保持不变) ---
        self.index_head = nn.Linear(self.hidden_size, 1, bias=False)  # 预测编辑位置
        self.edit_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)  # 预测编辑 token

        # --- 初始化 edit_head (保持不变) ---
        self._initialize_edit_head()

    def _initialize_edit_head(self):
        """使用 backbone 的 lm_head 初始化 edit_head"""
        lm_head = self.backbone.get_output_embeddings()  # 更通用的获取方式
        if lm_head is not None and isinstance(lm_head, nn.Linear):
            print("Initializing edit_head with lm_head weights...")
            # 确保权重形状匹配
            if self.edit_head.weight.shape == lm_head.weight.shape:
                self.edit_head.weight.data.copy_(lm_head.weight.data)
                # 处理 bias (如果存在且匹配)
                if hasattr(lm_head, 'bias') and lm_head.bias is not None:
                    if hasattr(self.edit_head, 'bias') and self.edit_head.bias is not None:
                        if self.edit_head.bias.shape == lm_head.bias.shape:
                            self.edit_head.bias.data.copy_(lm_head.bias.data)
                            print("Copied lm_head bias to edit_head.")
                        else:
                            print("Warning: lm_head bias shape mismatch, bias not copied.")
                    else:
                        print("Warning: lm_head has bias, but edit_head does not. Bias not copied.")
                print("edit_head weights initialized from lm_head.")
            else:
                print(
                    f"Warning: edit_head shape {self.edit_head.weight.shape} mismatch with lm_head shape {lm_head.weight.shape}. edit_head remains randomly initialized.")

        elif self.backbone.config.tie_word_embeddings and hasattr(self.backbone, 'get_input_embeddings'):
            # 处理权重绑定(tie_word_embeddings)的情况
            input_embeds = self.backbone.get_input_embeddings()
            if input_embeds is not None and self.edit_head.weight.shape == input_embeds.weight.shape:
                print("Initializing edit_head with input embedding weights (due to tied weights)...")
                self.edit_head.weight.data.copy_(input_embeds.weight.data)
                # 通常绑定权重时没有 bias
                if hasattr(self.edit_head, 'bias') and self.edit_head.bias is not None:
                    print(
                        "Warning: edit_head has bias, but weights are tied (usually no bias). Bias remains initialized.")
                print("edit_head weights initialized from input embeddings.")
            else:
                print(
                    f"Warning: edit_head shape {self.edit_head.weight.shape} mismatch with input embedding shape {input_embeds.weight.shape if input_embeds else 'N/A'}. edit_head remains randomly initialized.")
        else:
            print(
                "Warning: Could not find suitable lm_head or tied input embeddings in the backbone. edit_head remains randomly initialized.")

    def _build_gap_state(self, hidden_states: torch.Tensor,
                         attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        使用 GapEncoder 构建 gap 表示。

        Args:
            hidden_states: Backbone 最后一层输出 [B, L, D]
            attention_mask: Backbone 的注意力 mask [B, L] (1 表示有效 token, 0 表示 padding)

        Returns:
            Fused gap representation [B, L+1, D]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # 1. 创建 GAP 占位符 (使用 0 向量初始化)
        # 我们需要 L+1 个 GAP 位置
        gap_placeholder = torch.zeros((batch_size, 1, hidden_dim), device=device, dtype=dtype)

        # 2. 构建交错序列 [gap0, tok0, gap1, tok1, ..., gap(L-1), tok(L-1), gapL]
        # 目标长度: 2 * L + 1
        # 创建 L 个 gap (用于 token 之间)
        expanded_gaps = gap_placeholder.expand(batch_size, seq_len, hidden_dim)  # [B, L, D]

        # 使用 stack 和 reshape 进行交错
        # stacked: [B, L, 2, D] -> [[g0, t0], [g1, t1], ...]
        stacked = torch.stack([expanded_gaps, hidden_states], dim=2)

        # interleaved_part: [B, 2L, D] -> [g0, t0, g1, t1, ...]
        interleaved_part = stacked.view(batch_size, 2 * seq_len, hidden_dim)

        # 添加最后一个 gap (gapL)
        # final_gap: [B, 1, D]
        final_gap = gap_placeholder
        interleaved_sequence = torch.cat([interleaved_part, final_gap], dim=1)  # [B, 2L+1, D]

        # 3. 构建 GapEncoder 的 padding mask
        # Backbone 的 attention_mask [B, L] (1=有效, 0=pad)
        # 我们需要为交错序列 [B, 2L+1] 创建 mask
        # GAP 位置总是有效的 (mask=1)，Token 位置的有效性取决于原始 attention_mask
        encoder_mask = None
        if attention_mask is not None:
            # GAP mask: 全 1 [B, L+1]
            gap_mask = torch.ones((batch_size, seq_len + 1), device=device, dtype=torch.bool)  # True = 有效
            # Token mask: [B, L]
            token_mask = attention_mask.bool()  # 转换为 bool

            # 交错 mask
            # stacked_mask: [B, L, 2] -> [[True, mask0], [True, mask1], ...]
            stacked_mask = torch.stack([gap_mask[:, :-1], token_mask], dim=2)  # 使用前 L 个 gap mask
            # interleaved_mask_part: [B, 2L] -> [True, mask0, True, mask1, ...]
            interleaved_mask_part = stacked_mask.view(batch_size, 2 * seq_len)
            # 添加最后一个 gap 的 mask (总是 True)
            final_gap_mask = gap_mask[:, -1:]  # [B, 1]
            full_mask = torch.cat([interleaved_mask_part, final_gap_mask], dim=1)  # [B, 2L+1]

            # TransformerEncoder 需要 padding mask (True 表示 *无效* 位置)
            # 所以我们需要反转 full_mask
            encoder_mask = ~full_mask  # [B, 2L+1], True 表示 padding/无效

        # 4. 通过 GapEncoder 处理
        # encoded_sequence: [B, 2L+1, D]
        encoded_sequence = self.gap_encoder(interleaved_sequence, src_key_padding_mask=encoder_mask)

        # 5. 提取 GAP 位置的表示 (索引 0, 2, 4, ..., 2L)
        # gap_state: [B, L+1, D]
        gap_state = encoded_sequence[:, 0::2, :]

        return gap_state

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                target_index: Optional[torch.Tensor] = None,
                target_token: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        """
        前向传播：
          - 使用 GapEncoder 生成 gap state。
          - 训练模式：使用 edit_head 计算 token loss。
          - 推理模式：使用 edit_head 预测 token (不再需要 lm_head 回退)。
        """
        batch_size, seq_len = input_ids.shape

        # 1. 获取 backbone 的输出 (只需要最后一层 hidden states)
        # 注意：确保 backbone 配置为输出 hidden_states
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,  # 明确请求 hidden states
            return_dict=True
        )

        # 兼容不同 HF 模型输出格式
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            # 通常 hidden_states 是一个元组，最后一项是最后一层的输出
            hidden_states = outputs.hidden_states[-1]  # [B, L, D]
        elif hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state  # [B, L, D]
        else:
            # 尝试从 BaseModelOutput 获取
            try:
                # 对于某些模型（如纯粹的 encoder-decoder），可能需要访问 encoder 或 decoder 的输出
                if hasattr(self.backbone, 'encoder') and hasattr(outputs, 'encoder_last_hidden_state'):
                    hidden_states = outputs.encoder_last_hidden_state
                elif hasattr(self.backbone, 'decoder') and hasattr(outputs, 'decoder_last_hidden_state'):
                    hidden_states = outputs.decoder_last_hidden_state  # 可能需要调整，取决于模型类型
                else:
                    # 尝试直接从 outputs 字典获取
                    hidden_states = outputs['last_hidden_state']  # 常见 key
            except (AttributeError, KeyError, TypeError) as e:
                raise AttributeError(
                    f"Could not find last hidden states in backbone output. Output keys: {outputs.keys()}. Error: {e}")

        # 2. 构造融合了局部与全局信息的 gap 表示 (通过 GapEncoder)
        # 需要传入 attention_mask 来构建 GapEncoder 的 mask
        fused_gap_state = self._build_gap_state(hidden_states, attention_mask=attention_mask)  # [B, L+1, D]

        # 3. 处理推理和训练两种模式
        if target_index is None:
            # --- 推理模式 ---
            # a. 预测编辑位置
            idx_logits = self.index_head(fused_gap_state).squeeze(-1)  # [B, L+1]
            pred_index = idx_logits.argmax(-1)  # [B]

            # b. 根据预测的位置 gather gap state
            gathered_gap_state = torch.gather(
                fused_gap_state,  # [B, L+1, D]
                dim=1,
                index=pred_index.view(-1, 1, 1).expand(-1, 1, self.hidden_size)
            ).squeeze(1)  # [B, D]

            # c. 使用 edit_head 预测 token (不再需要 lm_head 回退)
            token_logits = self.edit_head(gathered_gap_state)  # [B, V]

            return dict(
                index_logits=idx_logits,
                token_logits=token_logits,
                pred_index=pred_index
            )
        else:
            # --- 训练模式 ---
            # a. 计算编辑位置预测的 logits 和 loss
            idx_logits = self.index_head(fused_gap_state).squeeze(-1)  # [B, L+1]
            idx_loss = F.cross_entropy(idx_logits, target_index)

            # b. 根据 *目标* 编辑位置 gather gap state
            gathered_gap_state = torch.gather(
                fused_gap_state,  # [B, L+1, D]
                dim=1,
                index=target_index.view(-1, 1, 1).expand(-1, 1, self.hidden_size)
            ).squeeze(1)  # [B, D]

            # c. 使用 edit_head 计算 token logits 和 loss
            token_logits = self.edit_head(gathered_gap_state)  # [B, V]
            tok_loss = F.cross_entropy(token_logits, target_token)

            # d. 计算总 loss
            loss = tok_loss + self.index_loss_weight * idx_loss

            return dict(
                loss=loss,
                tok_loss=tok_loss,
                idx_loss=idx_loss,
                index_logits=idx_logits,
                token_logits=token_logits
            )
