import warnings  # To warn if attention layers aren't found
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PretrainedConfig


class EditLMHF(nn.Module):
    """
    Improved EditLMHF that combines local context and global information for text editing:
    1. Uses [left, GAP, right] structure for local representation
    2. Leverages backbone's last-layer attention projections for global context
    3. Preserves original LM head for sequence-end predictions
    4. Uses fused local-global representations for in-sequence editing
    """

    def __init__(self,
                 base_model: str = "Qwen/Qwen2.5-0.5B",
                 index_loss_weight: float = 1.0,
                 freeze_shared_attn: bool = False):  # Option to freeze shared weights
        super().__init__()

        self.backbone = AutoModelForCausalLM.from_pretrained(
            base_model,
            low_cpu_mem_usage=True,
        )
        self.backbone.requires_grad_(True)

        # Basic attributes
        config = self.backbone.config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.index_loss_weight = index_loss_weight

        # Learnable embeddings
        self.gap_token_embed = nn.Parameter(torch.randn(1, 1, self.hidden_size) * 0.02)  # GAP token embedding
        self.boundary_embed = nn.Parameter(torch.zeros(1, 1, self.hidden_size))  # Boundary padding

        # Local triple projection (left, gap, right)
        self.triple_proj = nn.Linear(3 * self.hidden_size, self.hidden_size, bias=False)

        # Share backbone's attention projections
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = None, None, None, None
        # --- MODIFIED ---
        self.num_heads = None  # query heads
        self.num_kv_heads = None  # key / value heads (GQA / MQA)
        # --- END MODIFIED ---
        self._find_and_share_attn_projections(config, freeze_shared_attn)

        # Fusion layer for local and global contexts
        self.fuse_proj = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)

        # Output heads
        self.index_head = nn.Linear(self.hidden_size, 1, bias=False)  # Predicts edit position
        self.edit_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)  # Predicts edited token
        # --- Initialize edit_head with lm_head weights ---
        lm_head = self.backbone.lm_head if hasattr(self.backbone, 'lm_head') else None
        if lm_head is not None:
            print("Initializing edit_head with lm_head weights...")
            self.edit_head.weight.data.copy_(lm_head.weight.data)
            # Handle bias if lm_head has it (many don't)
            if hasattr(lm_head, 'bias') and lm_head.bias is not None:
                if hasattr(self.edit_head, 'bias') and self.edit_head.bias is not None:
                    self.edit_head.bias.data.copy_(lm_head.bias.data)
                    print("Copied lm_head bias to edit_head.")
                else:
                    print("Warning: lm_head has bias, but edit_head does not. Bias not copied.")
            print("edit_head initialized.")
        else:
            print("Warning: Could not find lm_head in the backbone. edit_head remains randomly initialized.")

    def _find_and_share_attn_projections(self, config: PretrainedConfig, freeze: bool):
        """
        Find and share attention projection layers from the backbone's last layer.
        Handles GQA/MQA by storing separate query and key/value head countsnew_params.
        """
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = None, None, None, None
        # --- MODIFIED ---
        self.num_heads = None  # query heads
        self.num_kv_heads = None  # key / value heads (GQA / MQA)
        # --- END MODIFIED ---

        # Get model type or infer from class name
        model_type = getattr(config, "model_type", None)
        if model_type is None:
            warnings.warn("Could not determine model_type from config. Attempting to infer.")
            model_class_name = self.backbone.__class__.__name__.lower()
            if "opt" in model_class_name:
                model_type = "opt"
            elif "llama" in model_class_name:
                model_type = "llama"
            elif "qwen2" in model_class_name:
                model_type = "qwen2"
            else:
                warnings.warn(f"Could not infer model_type. Aborting projection sharing.")
                return
        else:
            model_type = model_type.lower()

        print(f"Finding attention projections for model type: {model_type}")
        last_layer = None
        attn_module = None

        try:
            # Find last layer and attention module based on model architecture
            # (Existing logic for finding last_layer and attn_module remains the same)
            if model_type == "opt":
                decoder = getattr(self.backbone, "model", None)
                layers_module = getattr(decoder, "decoder", None)
                layers = getattr(layers_module, "layers", None)
                if layers:
                    last_layer = layers[-1]
                    attn_module = getattr(last_layer, "self_attn", None)
            elif model_type in ["llama", "mistral", "gemma", "qwen2"]:
                model = getattr(self.backbone, "model", None)
                layers = getattr(model, "layers", None)
                if layers:
                    last_layer = layers[-1]
                    attn_module = getattr(last_layer, "self_attn", None)
            elif model_type == "gpt_neox":
                transformer = getattr(self.backbone, "transformer", None)
                layers = getattr(transformer, "h", None)
                if layers:
                    last_layer = layers[-1]
                    attn_module = getattr(last_layer, "attention", None)
            elif model_type == "gpt2":
                transformer = getattr(self.backbone, "transformer", None)
                layers = getattr(transformer, "h", None)
                if layers:
                    last_layer = layers[-1]
                    attn_module = getattr(last_layer, "attn", None)
            elif model_type == "bloom":
                transformer = getattr(self.backbone, "transformer", None)
                layers = getattr(transformer, "h", None)
                if layers:
                    last_layer = layers[-1]
                    attn_module = getattr(last_layer, "self_attention", None)
            else:
                # Generic attempt for unsupported model types
                warnings.warn(f"Model type '{model_type}' not explicitly supported. Trying generic access.")
                model_or_transformer = getattr(self.backbone, "model", getattr(self.backbone, "transformer", None))
                layers = getattr(model_or_transformer, "layers", getattr(model_or_transformer, "h", None))
                if layers:
                    last_layer = layers[-1]
                    attn_module = getattr(last_layer, "self_attn",
                                          getattr(last_layer, "attention", getattr(last_layer, "attn", None)))

            # Get projection layers if attention module was found
            if attn_module:
                print(f"Found attention module: {type(attn_module)}")

                # Try to get separate Q, K, V projections
                q_proj = getattr(attn_module, "q_proj", None)
                k_proj = getattr(attn_module, "k_proj", None)
                v_proj = getattr(attn_module, "v_proj", None)
                o_proj = getattr(attn_module, "o_proj",
                                 getattr(attn_module, "dense", getattr(attn_module, "out_proj", None)))

                # --- MODIFIED: Get head counts ---
                # ❶ 读取 query / kv 头数
                self.num_heads = getattr(config, "num_attention_heads", None)
                self.num_kv_heads = getattr(attn_module, "num_key_value_heads",
                                            getattr(config, "num_key_value_heads",
                                                    getattr(attn_module, "num_heads",
                                                            None)))  # Fallback to attn_module.num_heads

                if self.num_kv_heads is None and self.num_heads is not None:  # 非 GQA/MQA 模型 (or info missing)
                    print("KV heads count not found, assuming equal to query heads (MHA).")
                    self.num_kv_heads = self.num_heads
                # --- END MODIFIED ---

                if all([q_proj, k_proj, v_proj, o_proj]):
                    # Found all separate projections
                    self.q_proj, self.k_proj, self.v_proj, self.o_proj = q_proj, k_proj, v_proj, o_proj
                    print(f"Found separate Q/K/V/O projections for {model_type}.")
                    # Removed head count assignment here, handled above

                # Handle special cases with combined QKV projections
                elif model_type == "gpt2" and hasattr(attn_module, 'c_attn') and hasattr(attn_module, 'c_proj'):
                    warnings.warn(f"Model uses combined QKV projection. Only sharing output projection.")
                    self.o_proj = getattr(attn_module, 'c_proj', None)
                    # Head counts already assigned above

                elif model_type == "bloom" and hasattr(attn_module, "query_key_value") and hasattr(attn_module,
                                                                                                   "dense"):
                    warnings.warn(f"Model uses combined query_key_value. Only sharing output projection.")
                    self.o_proj = getattr(attn_module, 'dense', None)
                    # Head counts already assigned above

                else:
                    # Check for other combined projection patterns
                    combined_qkv = getattr(attn_module, "query_key_value", getattr(attn_module, "c_attn", None))
                    output_p = getattr(attn_module, "dense",
                                       getattr(attn_module, "o_proj", getattr(attn_module, "c_proj", None)))

                    if combined_qkv and output_p:
                        warnings.warn(f"Found combined QKV projection. Only sharing output projection.")
                        self.o_proj = output_p
                        # Head counts already assigned above
                    else:
                        warnings.warn(f"Could not find required projection layers. Sharing failed.")

            else:
                warnings.warn(f"Could not locate attention module. Auto-sharing failed.")
                # Try to get head counts from config even if module not found
                self.num_heads = getattr(config, "num_attention_heads", None)
                self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)  # Fallback to query heads

        except Exception as e:
            warnings.warn(f"Error during attention projection sharing: {e}")
            # Try to get head counts from config as a last resort
            if self.num_heads is None:
                self.num_heads = getattr(config, "num_attention_heads", None)
            if self.num_kv_heads is None:
                self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)  # Fallback to query heads

        # Final check and freezing
        if not all([self.q_proj, self.k_proj, self.v_proj, self.o_proj]):
            warnings.warn("Failed to find all projection layers. Global context will be disabled.")
            self.q_proj, self.k_proj, self.v_proj, self.o_proj = None, None, None, None
        # --- MODIFIED ---
        elif self.num_heads is None:
            warnings.warn("Could not determine number of query heads.")
        elif self.num_kv_heads is None:
            warnings.warn("Could not determine number of key/value heads.")
        # --- END MODIFIED ---
        else:
            print(
                f"Successfully shared attention projections (Query Heads: {self.num_heads}, KV Heads: {self.num_kv_heads}).")
            if freeze:
                for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
                    if proj:
                        proj.requires_grad_(False)
                print("Attention projections frozen.")

    def _build_gap_state(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Build representation for each possible gap position, combining local and global context.
        Handles GQA/MQA correctly.

        Args:
            hidden_states: Last layer hidden states [B, L, D]

        Returns:
            Fused gap representation [B, L+1, D]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        dtype = hidden_states.dtype

        # 1. Build local triplet representation [left, gap, right]
        padding = self.boundary_embed.expand(batch_size, -1, -1).to(dtype=dtype)

        # Create left, right and center contexts
        left_context = torch.cat([padding, hidden_states], dim=1)  # [B, L+1, D]
        right_context = torch.cat([hidden_states, padding], dim=1)  # [B, L+1, D]
        center_context = self.gap_token_embed.expand(
            batch_size, seq_len + 1, hidden_dim).to(dtype=dtype)  # [B, L+1, D]

        # Combine and project local representation
        local_triple = torch.cat([left_context, center_context, right_context], dim=-1)  # [B, L+1, 3*D]
        gap_local = self.triple_proj(local_triple)  # [B, L+1, D]

        # 2. Calculate global context if attention projections are available
        # --- MODIFIED: Check for num_kv_heads as well ---
        if not all([self.q_proj, self.k_proj, self.v_proj, self.o_proj, self.num_heads, self.num_kv_heads]):
            # Fall back to zeros if global context unavailable
            raise RuntimeError("Global context disabled due to missing projections or head counts.")
        else:
            # --- MODIFIED: GQA/MQA Handling ---
            if hidden_dim % self.num_heads != 0:
                raise ValueError(
                    f"hidden_size {hidden_dim} must be divisible by num_heads (query heads) {self.num_heads}")
            head_dim = hidden_dim // self.num_heads  # Calculate head_dim based on query heads

            # Project queries, keys and values
            query = self.q_proj(gap_local)  # [B, L+1, D] (D = num_heads * head_dim)
            key = self.k_proj(hidden_states)  # [B, L, D_kv] (D_kv = num_kv_heads * head_dim)
            value = self.v_proj(hidden_states)  # [B, L, D_kv]

            # Reshape for multi-head attention
            def split_heads(tensor, num_target_heads, head_dim):
                """Splits hidden_size into num_heads x head_dim"""
                B, S, D_tensor = tensor.shape
                # Ensure tensor dimension matches expected num_heads * head_dim
                if D_tensor != num_target_heads * head_dim:
                    raise ValueError(
                        f"Tensor dimension {D_tensor} does not match num_target_heads {num_target_heads} * head_dim {head_dim}")
                return tensor.view(B, S, num_target_heads, head_dim).transpose(1, 2)

            # Split heads using respective head counts
            query_split = split_heads(query, self.num_heads, head_dim)  # [B, H_q, L+1, d]
            key_split = split_heads(key, self.num_kv_heads, head_dim)  # [B, H_kv, L, d]
            value_split = split_heads(value, self.num_kv_heads, head_dim)  # [B, H_kv, L, d]

            # Handle GQA/MQA: Repeat K/V heads if necessary
            if self.num_kv_heads != self.num_heads:
                if self.num_heads % self.num_kv_heads != 0:
                    raise ValueError(
                        f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads}) for GQA/MQA")
                repeat_factor = self.num_heads // self.num_kv_heads
                # print(f"Applying GQA/MQA: Repeating K/V heads by {repeat_factor}")
                key_split = key_split.repeat_interleave(repeat_factor, dim=1)  # [B, H_q, L, d]
                value_split = value_split.repeat_interleave(repeat_factor, dim=1)  # [B, H_q, L, d]
            # --- END MODIFIED ---

            # Apply scaled dot-product attention
            # scaled_dot_product_attention handles broadcasting if shapes match after repeat_interleave
            attn_output = F.scaled_dot_product_attention(
                query_split, key_split, value_split,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False  # We want the GAP state to attend to all context tokens
            )  # [B, H_q, L+1, d]

            # Reshape and project output
            attn_output = attn_output.transpose(1, 2).contiguous().view(
                batch_size, seq_len + 1, hidden_dim)  # [B, L+1, D]
            global_ctx = self.o_proj(attn_output)  # [B, L+1, D]

        # 3. Fuse local and global representations
        fused_input = torch.cat([gap_local, global_ctx], dim=-1)  # [B, L+1, 2*D]
        fused_gap_state = self.fuse_proj(fused_input)  # [B, L+1, D]

        return fused_gap_state

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                target_index: Optional[torch.Tensor] = None,
                target_token: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        """
        前向传播：
          - 训练模式：无论 target_index 是多少，都通过 edit_head 输出来计算 token loss
          - 推理模式：若预测位置等于 seq_len，则使用 lm_head 的输出作为 token 预测结果
        """
        batch_size, seq_len = input_ids.shape

        # 1. 获取 backbone 的输出和语言模型 logits
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state  # [B, L, D]
        elif hasattr(outputs, 'hidden_states'):
            hidden_states = outputs.hidden_states[-1]  # [B, L, D]
        else:
            raise AttributeError("找不到 backbone 输出中的 hidden_states")

        if hasattr(outputs, 'logits'):
            lm_logits = outputs.logits  # [B, L, V]
        elif hasattr(self.backbone, 'lm_head'):
            lm_logits = self.backbone.lm_head(hidden_states)
        else:
            raise AttributeError("找不到 lm_logits 输出")

        # 2. 构造融合了局部与全局信息的 gap 表示
        fused_gap_state = self._build_gap_state(hidden_states)  # [B, L+1, D]

        # 3. 处理推理和训练两种模式
        if target_index is None:
            # 推理模式
            idx_logits = self.index_head(fused_gap_state).squeeze(-1)  # [B, L+1]
            # 预测编辑位置
            pred_index = idx_logits.argmax(-1)  # [B]
            # 获取 lm_head 输出备用（标准 LM 预测）
            lm_last_token = lm_logits[:, seq_len - 1, :]  # [B, V]
            # Gather the gap state 对应预测的位置
            gathered_gap_state = torch.gather(
                fused_gap_state,  # [B, L+1, D]
                dim=1,
                index=pred_index.view(-1, 1, 1).expand(-1, 1, self.hidden_size)
            ).squeeze(1)  # [B, D]
            # 使用 edit_head 预测 token
            gathered_edit = self.edit_head(gathered_gap_state)  # [B, V]

            # 如果预测位置等于 seq_len，则使用 lm_head 的输出
            # 生成一个布尔向量表示哪些样本满足条件
            use_lm_mask_scalar = torch.eq(pred_index, seq_len)  # [B]
            final_token_logits = torch.where(
                use_lm_mask_scalar.unsqueeze(-1),  # [B, 1]
                lm_last_token,  # [B, V]
                gathered_edit  # [B, V]
            )

            return dict(
                index_logits=idx_logits,
                token_logits=final_token_logits,
                pred_index=pred_index
            )
        else:
            # 训练模式：无论 target_index 为什么，都只用 edit_head 来进行 token loss 的计算

            idx_logits = self.index_head(fused_gap_state).squeeze(-1)  # [B, L+1]

            # Gather the gap state 对应目标编辑位置
            gathered_gap_state = torch.gather(
                fused_gap_state,  # [B, L+1, D]
                dim=1,
                index=target_index.view(-1, 1, 1).expand(-1, 1, self.hidden_size)
            ).squeeze(1)  # [B, D]

            # 总是通过 edit_head 计算 token 输出
            gathered_edit = self.edit_head(gathered_gap_state)  # [B, V]
            # 计算 token loss（交叉熵损失）
            tok_loss = F.cross_entropy(gathered_edit, target_token)
            # 计算编辑位置预测的 loss
            idx_loss = F.cross_entropy(idx_logits, target_index)
            loss = tok_loss + self.index_loss_weight * idx_loss

            return dict(
                loss=loss,
                tok_loss=tok_loss,
                idx_loss=idx_loss,
                index_logits=idx_logits,
                token_logits=gathered_edit
            )
