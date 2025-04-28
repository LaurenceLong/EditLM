import warnings
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PretrainedConfig


class EditLMHF(nn.Module):
    """
    Improved EditLMHF model that enhances the original with several architectural improvements:
    1. Uses MLP for triple_proj and fuse_proj instead of simple linear layers (Suggestion 1)
    2. Applies LayerNorm before heads and Dropout before edit_head (Suggestion 3)
    3. Initializes boundary_embed with small random noise for better stability (Suggestion 5)
    4. Adds residual connections around MLPs to improve gradient flow (Suggestion 6)
    """

    def __init__(self,
                 base_model: str = "Qwen/Qwen2.5-0.5B",
                 index_loss_weight: float = 1.0,
                 freeze_shared_attn: bool = False,
                 edit_dropout_prob: float = 0.1):  # Added dropout probability configuration
        super().__init__()

        # Load pre-trained language model
        self.backbone = AutoModelForCausalLM.from_pretrained(
            base_model,
            low_cpu_mem_usage=True,
        )
        self.backbone.requires_grad_(True)  # Enable gradient updates for backbone

        # Extract basic configuration from the backbone model
        config = self.backbone.config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.index_loss_weight = index_loss_weight  # Weight for index prediction loss

        # Learnable embeddings for special tokens
        # Gap token represents the insertion point in the text
        self.gap_token_embed = nn.Parameter(torch.randn(1, 1, self.hidden_size) * 0.02)
        # Boundary embed represents text boundaries with small noise initialization (Suggestion 5)
        self.boundary_embed = nn.Parameter(torch.randn(1, 1, self.hidden_size) * 0.02)

        # MLP for processing local triple context (left, gap, right) - Suggestion 1
        self.triple_mlp = nn.Sequential(
            nn.Linear(3 * self.hidden_size, self.hidden_size),
            nn.GELU(),  # Non-linear activation
            nn.LayerNorm(self.hidden_size),  # Normalization for stability
        )

        # MLP for fusing local and global contexts - Suggestion 1
        self.fuse_mlp = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.GELU(),  # Non-linear activation
            nn.LayerNorm(self.hidden_size),  # Normalization for stability
        )

        # Initialize attention projection references as None
        # These will be filled with shared weights from the backbone
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = None, None, None, None
        self.num_heads = None
        self.num_kv_heads = None
        self._find_and_share_attn_projections(config, freeze_shared_attn)

        # Add LayerNorm before heads and Dropout before edit_head (Suggestion 3)
        self.pre_head_ln = nn.LayerNorm(self.hidden_size)
        self.edit_dropout = nn.Dropout(edit_dropout_prob)

        # Output prediction heads
        self.index_head = nn.Linear(self.hidden_size, 1, bias=False)  # Predicts insertion point
        self.edit_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)  # Predicts inserted token

        # Initialize edit_head with lm_head weights for better starting point
        lm_head = self.backbone.lm_head if hasattr(self.backbone, 'lm_head') else None
        if lm_head is not None:
            print("Initializing edit_head with lm_head weights...")
            self.edit_head.weight.data.copy_(lm_head.weight.data)
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
        This enables the model to reuse the transformer's attention mechanism.

        The function handles Grouped Query Attention (GQA) and Multi-Query Attention (MQA)
        by tracking separate head counts for queries and key/values.

        Args:
            config: Model configuration
            freeze: Whether to freeze the shared attention projections
        """
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = None, None, None, None
        self.num_heads = None
        self.num_kv_heads = None

        # Determine model type from config or class name
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
            # Access model architecture based on model type to find attention modules
            # Different model families organize their layers differently
            if model_type == "opt":
                layers = getattr(getattr(getattr(self.backbone, "model", None), "decoder", None), "layers", None)
                if layers: last_layer = layers[-1]; attn_module = getattr(last_layer, "self_attn", None)
            elif model_type in ["llama", "mistral", "gemma", "qwen2"]:
                layers = getattr(getattr(self.backbone, "model", None), "layers", None)
                if layers: last_layer = layers[-1]; attn_module = getattr(last_layer, "self_attn", None)
            elif model_type == "gpt_neox":
                layers = getattr(getattr(self.backbone, "transformer", None), "h", None)
                if layers: last_layer = layers[-1]; attn_module = getattr(last_layer, "attention", None)
            elif model_type == "gpt2":
                layers = getattr(getattr(self.backbone, "transformer", None), "h", None)
                if layers: last_layer = layers[-1]; attn_module = getattr(last_layer, "attn", None)
            elif model_type == "bloom":
                layers = getattr(getattr(self.backbone, "transformer", None), "h", None)
                if layers: last_layer = layers[-1]; attn_module = getattr(last_layer, "self_attention", None)
            else:
                # Generic fallback for unknown model types
                warnings.warn(f"Model type '{model_type}' not explicitly supported. Trying generic access.")
                model_or_transformer = getattr(self.backbone, "model", getattr(self.backbone, "transformer", None))
                layers = getattr(model_or_transformer, "layers", getattr(model_or_transformer, "h", None))
                if layers:
                    last_layer = layers[-1]
                    attn_module = getattr(last_layer, "self_attn",
                                          getattr(last_layer, "attention", getattr(last_layer, "attn", None)))

            # Extract attention projection modules if found
            if attn_module:
                print(f"Found attention module: {type(attn_module)}")
                q_proj = getattr(attn_module, "q_proj", None)
                k_proj = getattr(attn_module, "k_proj", None)
                v_proj = getattr(attn_module, "v_proj", None)
                o_proj = getattr(attn_module, "o_proj",
                                 getattr(attn_module, "dense", getattr(attn_module, "out_proj", None)))

                # Determine number of attention heads
                self.num_heads = getattr(config, "num_attention_heads", None)
                self.num_kv_heads = getattr(attn_module, "num_key_value_heads", getattr(config, "num_key_value_heads",
                                                                                        getattr(attn_module,
                                                                                                "num_heads", None)))
                if self.num_kv_heads is None and self.num_heads is not None:
                    print("KV heads count not found, assuming equal to query heads (MHA).")
                    self.num_kv_heads = self.num_heads

                # Handle models with separate Q/K/V projections
                if all([q_proj, k_proj, v_proj, o_proj]):
                    self.q_proj, self.k_proj, self.v_proj, self.o_proj = q_proj, k_proj, v_proj, o_proj
                    print(f"Found separate Q/K/V/O projections for {model_type}.")
                # Handle models with combined QKV projections (e.g., GPT-2, Bloom)
                elif model_type == "gpt2" and hasattr(attn_module, 'c_attn') and hasattr(attn_module, 'c_proj'):
                    warnings.warn(f"Model uses combined QKV projection. Only sharing output projection.")
                    self.o_proj = getattr(attn_module, 'c_proj', None)
                elif model_type == "bloom" and hasattr(attn_module, "query_key_value") and hasattr(attn_module,
                                                                                                   "dense"):
                    warnings.warn(f"Model uses combined query_key_value. Only sharing output projection.")
                    self.o_proj = getattr(attn_module, 'dense', None)
                else:
                    # Generic fallback for other combined projection patterns
                    combined_qkv = getattr(attn_module, "query_key_value", getattr(attn_module, "c_attn", None))
                    output_p = getattr(attn_module, "dense",
                                       getattr(attn_module, "o_proj", getattr(attn_module, "c_proj", None)))
                    if combined_qkv and output_p:
                        warnings.warn(f"Found combined QKV projection. Only sharing output projection.")
                        self.o_proj = output_p
                    else:
                        warnings.warn(f"Could not find required projection layers. Sharing failed.")
            else:
                warnings.warn(f"Could not locate attention module. Auto-sharing failed.")
                self.num_heads = getattr(config, "num_attention_heads", None)
                self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)

        except Exception as e:
            warnings.warn(f"Error during attention projection sharing: {e}")
            if self.num_heads is None: self.num_heads = getattr(config, "num_attention_heads", None)
            if self.num_kv_heads is None: self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)

        # Final validation and optional freezing
        if not all([self.q_proj, self.k_proj, self.v_proj, self.o_proj]):
            warnings.warn("Failed to find all projection layers. Global context will be disabled.")
            self.q_proj, self.k_proj, self.v_proj, self.o_proj = None, None, None, None
        elif self.num_heads is None:
            warnings.warn("Could not determine number of query heads.")
        elif self.num_kv_heads is None:
            warnings.warn("Could not determine number of key/value heads.")
        else:
            print(
                f"Successfully shared attention projections (Query Heads: {self.num_heads}, KV Heads: {self.num_kv_heads}).")
            if freeze:
                for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
                    if proj: proj.requires_grad_(False)
                print("Attention projections frozen.")

    def _build_gap_state(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Build representation for each possible gap position by combining local context
        (left token, gap token, right token) with global context (attention).

        Uses MLPs with residual connections to fuse information.

        Args:
            hidden_states: Last layer hidden states [batch_size, seq_len, hidden_dim]

        Returns:
            Fused gap representation [batch_size, seq_len+1, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        dtype = hidden_states.dtype

        # 1. Build local triplet representation [left, gap, right]
        # Expand boundary_embed to match batch size and add as padding
        padding = self.boundary_embed.expand(batch_size, -1, -1).to(dtype=dtype)  # [B, 1, D]

        # Create left and right context by shifting with padding
        left_context = torch.cat([padding, hidden_states], dim=1)  # [B, L+1, D]
        right_context = torch.cat([hidden_states, padding], dim=1)  # [B, L+1, D]
        # Create gap token representation for each position
        center_context = self.gap_token_embed.expand(
            batch_size, seq_len + 1, hidden_dim).to(dtype=dtype)  # [B, L+1, D]

        # Combine into local triple context
        local_triple = torch.cat([left_context, center_context, right_context], dim=-1)  # [B, L+1, 3*D]

        # Process with MLP and add residual connection (Suggestion 1 & 6)
        gap_local = self.triple_mlp(local_triple) + self.boundary_embed.expand(batch_size, seq_len + 1, -1).to(
            dtype=dtype)  # [B, L+1, D]

        # 2. Calculate global context using backbone's attention mechanism
        global_ctx = None  # Initialize global context
        if all([self.q_proj, self.k_proj, self.v_proj, self.o_proj, self.num_heads, self.num_kv_heads]):
            try:
                # Validate dimensions
                if hidden_dim % self.num_heads != 0:
                    raise ValueError(f"hidden_size {hidden_dim} must be divisible by num_heads {self.num_heads}")
                head_dim = hidden_dim // self.num_heads

                # Project query (from gap positions) and key/value (from sequence)
                query = self.q_proj(gap_local)  # [B, L+1, D]
                key = self.k_proj(hidden_states)  # [B, L, D_kv]
                value = self.v_proj(hidden_states)  # [B, L, D_kv]

                # Helper function to reshape tensors for multi-head attention
                def split_heads(tensor, num_target_heads, head_dim):
                    B, S, D_tensor = tensor.shape
                    if D_tensor != num_target_heads * head_dim:
                        raise ValueError(f"Tensor dim {D_tensor} != {num_target_heads} * {head_dim}")
                    return tensor.view(B, S, num_target_heads, head_dim).transpose(1, 2)

                # Split heads for multi-head attention
                query_split = split_heads(query, self.num_heads, head_dim)  # [B, H_q, L+1, d]
                key_split = split_heads(key, self.num_kv_heads, head_dim)  # [B, H_kv, L, d]
                value_split = split_heads(value, self.num_kv_heads, head_dim)  # [B, H_kv, L, d]

                # Handle Grouped Query Attention (GQA) or Multi-Query Attention (MQA)
                if self.num_kv_heads != self.num_heads:
                    if self.num_heads % self.num_kv_heads != 0:
                        raise ValueError(
                            f"num_heads {self.num_heads} not divisible by num_kv_heads {self.num_kv_heads}")
                    repeat_factor = self.num_heads // self.num_kv_heads
                    # Repeat keys and values to match query head count
                    key_split = key_split.repeat_interleave(repeat_factor, dim=1)  # [B, H_q, L, d]
                    value_split = value_split.repeat_interleave(repeat_factor, dim=1)  # [B, H_q, L, d]

                # Compute attention (non-causal since we're attending to all tokens)
                attn_output = F.scaled_dot_product_attention(
                    query_split, key_split, value_split, is_causal=False
                )  # [B, H_q, L+1, d]

                # Reshape and project
                attn_output = attn_output.transpose(1, 2).contiguous().view(
                    batch_size, seq_len + 1, hidden_dim)  # [B, L+1, D]
                global_ctx = self.o_proj(attn_output)  # [B, L+1, D]

            except Exception as e:
                warnings.warn(f"Error during global context calculation: {e}. Global context disabled for this batch.")
                global_ctx = torch.zeros_like(gap_local)  # Fallback to zeros on error
        else:
            # If global context cannot be calculated (missing projections), use zeros
            warnings.warn("Global context disabled due to missing projections or head counts. Using zeros.")
            global_ctx = torch.zeros_like(gap_local)  # Fallback to zeros

        # 3. Fuse local and global representations using MLP with residual connection
        fused_input = torch.cat([gap_local, global_ctx], dim=-1)  # [B, L+1, 2*D]
        # Apply fusion MLP and add residual connection (Suggestion 1 & 6)
        fused_gap_state = self.fuse_mlp(fused_input) + gap_local  # [B, L+1, D]

        return fused_gap_state

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                target_index: Optional[torch.Tensor] = None,
                target_token: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask for padding [batch_size, seq_len]
            target_index: Target insertion indices for training [batch_size]
            target_token: Target tokens to insert for training [batch_size]

        Returns:
            Dictionary with model outputs (varies between training and inference)
        """
        batch_size, seq_len = input_ids.shape

        # 1. Run input through the backbone model
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # Extract hidden states and logits from backbone
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        elif hasattr(outputs, 'hidden_states'):
            hidden_states = outputs.hidden_states[-1]
        else:
            raise AttributeError("Cannot find hidden_states in backbone output")

        if hasattr(outputs, 'logits'):
            lm_logits = outputs.logits
        elif hasattr(self.backbone, 'lm_head'):
            lm_logits = self.backbone.lm_head(hidden_states)
        else:
            raise AttributeError("Cannot find lm_logits output")

        # 2. Build fused gap state (now uses MLPs with residuals)
        fused_gap_state = self._build_gap_state(hidden_states)  # [B, L+1, D]

        # 3. Process based on mode (inference vs. training)
        if target_index is None:
            # --- Inference Mode ---
            # Apply LayerNorm before index prediction
            normed_fused_gap_state = self.pre_head_ln(fused_gap_state)  # [B, L+1, D]
            idx_logits = self.index_head(normed_fused_gap_state).squeeze(-1)  # [B, L+1]

            # Find predicted insertion position
            pred_index = idx_logits.argmax(-1)  # [B]
            # Get last token prediction from backbone (used when no insertion needed)
            lm_last_token_logits = lm_logits[:, seq_len - 1, :]  # [B, V]

            # Gather the gap state corresponding to the predicted insertion point
            gathered_gap_state = torch.gather(
                fused_gap_state,  # Use original state for gathering
                dim=1,
                index=pred_index.view(-1, 1, 1).expand(-1, 1, self.hidden_size)
            ).squeeze(1)  # [B, D]

            # Apply LayerNorm and Dropout before edit head (Suggestion 3)
            normed_gathered_state = self.pre_head_ln(gathered_gap_state)  # [B, D]
            dropout_state = self.edit_dropout(normed_gathered_state)  # [B, D]
            gathered_edit_logits = self.edit_head(dropout_state)  # [B, V]

            # Use lm_head logits if predicted index is at the end (seq_len)
            # This handles cases where no insertion is needed
            use_lm_mask = torch.eq(pred_index, seq_len)  # [B]
            final_token_logits = torch.where(
                use_lm_mask.unsqueeze(-1),  # [B, 1]
                lm_last_token_logits,  # [B, V]
                gathered_edit_logits  # [B, V]
            )

            return dict(
                index_logits=idx_logits,
                token_logits=final_token_logits,
                pred_index=pred_index
            )
        else:
            # --- Training Mode ---
            # Apply LayerNorm before index head
            normed_fused_gap_state = self.pre_head_ln(fused_gap_state)  # [B, L+1, D]
            idx_logits = self.index_head(normed_fused_gap_state).squeeze(-1)  # [B, L+1]

            # Gather the gap state corresponding to the target insertion index
            gathered_gap_state = torch.gather(
                fused_gap_state,  # Use original state for gathering
                dim=1,
                index=target_index.view(-1, 1, 1).expand(-1, 1, self.hidden_size)
            ).squeeze(1)  # [B, D]

            # Apply LayerNorm and Dropout before edit head (Suggestion 3)
            normed_gathered_state = self.pre_head_ln(gathered_gap_state)  # [B, D]
            dropout_state = self.edit_dropout(normed_gathered_state)  # [B, D]
            gathered_edit_logits = self.edit_head(dropout_state)  # [B, V]

            # Calculate losses for both index prediction and token prediction
            tok_loss = F.cross_entropy(gathered_edit_logits, target_token)  # Token prediction loss
            idx_loss = F.cross_entropy(idx_logits, target_index)  # Index prediction loss
            loss = tok_loss + self.index_loss_weight * idx_loss  # Combined loss

            return dict(
                loss=loss,
                tok_loss=tok_loss,
                idx_loss=idx_loss,
                index_logits=idx_logits,  # Logits from normalized state
                token_logits=gathered_edit_logits  # Logits from normalized+dropout state
            )