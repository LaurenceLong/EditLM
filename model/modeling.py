import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedModel
from typing import Optional, Tuple, Dict, Any, Union


class EditLM(nn.Module):
    """
    极简(Index, Token)编辑模型，无⟨CURSOR⟩
    支持多任务预训练
    """

    def __init__(self, config, model_name_or_path: str, tokenizer_length: int):
        super().__init__()
        self.config = config

        # 加载基础模型
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

        # 调整词嵌入层和LM头以匹配新的词汇表大小
        self.resize_token_embeddings(tokenizer_length)

        # 获取隐藏层大小
        self.hidden_size = self.base_model.config.hidden_size

        # 添加索引预测头 (Position Scoring Head)
        # 对每个位置的隐藏状态进行线性变换，得到该位置作为编辑点的分数
        # 输出维度将在前向传播时动态确定为序列长度+1
        self.index_head = nn.Linear(self.hidden_size, 1)

        # 添加语言模型预测头用于多任务学习
        self.lm_prediction_head = nn.Linear(self.hidden_size, tokenizer_length)

        # 初始化权重
        self._init_weights(self.index_head)
        self._init_weights(self.lm_prediction_head)

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def resize_token_embeddings(self, new_size: int):
        """调整词嵌入层大小以适应新的词汇表"""
        self.base_model.resize_token_embeddings(new_size)

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            target_token_ids: Optional[torch.LongTensor] = None,
            target_index_positions: Optional[torch.LongTensor] = None,
            lm_targets: Optional[torch.LongTensor] = None,
            lm_positions: Optional[torch.LongTensor] = None,
            return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播，预测编辑索引和编辑词元，以及语言模型目标

        Args:
            input_ids: 输入序列，形状为[batch_size, seq_len]
            attention_mask: 注意力掩码，形状为[batch_size, seq_len]
            target_token_ids: 目标词元ID，形状为[batch_size]，用于计算损失
            target_index_positions: 目标索引位置，形状为[batch_size]，用于计算损失
            lm_targets: 语言模型预测目标，形状为[batch_size]
            lm_positions: 语言模型预测位置，形状为[batch_size]
            return_dict: 是否返回字典格式的输出

        Returns:
            包含词元预测、索引预测和语言模型预测的输出字典
        """
        # 通过基础模型获取隐藏状态
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )

        # 获取隐藏状态
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]

        # 1. 词元预测 (LM Head)
        # 直接使用基础模型的语言模型头
        token_logits_full = outputs.logits  # [batch_size, seq_len, vocab_size]

        # 对于词元预测，我们聚合整个序列的信息（取最后一个token的表示）
        # 注意：这可能是一个简化，对于中间编辑任务可能不是最优设计
        token_logits = token_logits_full[:, -1, :]  # [batch_size, vocab_size]

        # 2. 索引预测 (Position Scoring Head)
        # 将每个位置的隐藏状态通过线性层映射为一个分数，表示该位置作为编辑点的可能性
        batch_size, seq_len, _ = hidden_states.shape

        # 计算每个输入位置的分数
        position_scores = self.index_head(hidden_states)  # [batch_size, seq_len, 1]
        position_scores = position_scores.squeeze(-1)  # [batch_size, seq_len]

        # 增加一个额外位置表示序列末尾 (或者理解为在最后一个token之后插入)
        # 使用零作为这个额外位置的初始分数，模型可以通过学习调整其他位置的分数来相对地使其变高或变低
        pad = torch.zeros((batch_size, 1), device=position_scores.device, dtype=position_scores.dtype)
        index_logits = torch.cat([position_scores, pad], dim=1)  # [batch_size, seq_len+1]

        # 3. 语言模型预测 (对特定位置进行预测)
        # 为多任务学习提供额外的语言模型目标
        lm_logits = None
        if lm_positions is not None:
            # 提取指定位置的隐藏状态
            # 确保 lm_positions 在有效范围内 (0 <= pos < seq_len)
            if (lm_positions >= 0).all() and (lm_positions < seq_len).all():
                batch_indices = torch.arange(batch_size, device=hidden_states.device)
                selected_hidden = hidden_states[batch_indices, lm_positions]  # [batch_size, hidden_size]

                # 预测这些位置的词元
                lm_logits = self.lm_prediction_head(selected_hidden)  # [batch_size, vocab_size]
            else:
                # 如果 lm_positions 无效，可以选择报错或跳过计算
                print(
                    f"Warning: lm_positions contain invalid indices. Max index should be {seq_len - 1}. Skipping LM prediction for this batch.")
                # 或者根据需要处理，例如 lm_logits 保持为 None

        # 准备输出
        outputs_dict = {  # Renamed to avoid conflict with base model 'outputs'
            "token_logits": token_logits,
            "index_logits": index_logits,
            "lm_logits": lm_logits
        }

        # 如果提供了目标，计算损失
        lm_loss = None  # Initialize lm_loss

        if target_token_ids is not None and target_index_positions is not None:
            # --- 输入验证 (可选但推荐) ---
            # 验证 target_index_positions 是否在 [0, seq_len] 范围内
            if not ((target_index_positions >= 0) & (target_index_positions <= seq_len)).all():
                raise ValueError(
                    f"target_index_positions must be between 0 and {seq_len}, but got values outside this range.")
            # --- 结束验证 ---

            # 计算词元预测损失
            token_loss = nn.functional.cross_entropy(
                token_logits, target_token_ids
            )

            # 计算索引预测损失
            index_loss = nn.functional.cross_entropy(
                index_logits, target_index_positions
            )

            # 计算语言模型预测损失(如果提供了目标)
            current_lm_loss = torch.tensor(0.0, device=token_loss.device)  # Default to 0
            if lm_targets is not None and lm_logits is not None:
                # 确保 lm_targets 和 lm_logits 对应
                if lm_logits.shape[0] == lm_targets.shape[0]:  # Basic check
                    current_lm_loss = nn.functional.cross_entropy(
                        lm_logits, lm_targets,  # Make sure lm_targets are also correctly batched/indexed if needed
                        ignore_index=-100  # Common practice to ignore padding tokens if applicable
                    )
                else:
                    print(
                        "Warning: Mismatch between lm_logits and lm_targets batch size. Skipping LM loss calculation.")

            # 总损失 = 词元损失 + α * 索引损失 + β * 语言模型损失
            loss = token_loss + self.config.index_loss_weight * index_loss

            # 如果启用了语言模型预训练
            if hasattr(self.config, 'lm_loss_weight') and self.config.lm_loss_weight > 0 and lm_logits is not None:
                # Only add lm_loss if its weight is positive and it was computed
                loss += self.config.lm_loss_weight * current_lm_loss
                lm_loss = current_lm_loss  # Store the computed LM loss

            outputs_dict["loss"] = loss
            outputs_dict["token_loss"] = token_loss
            outputs_dict["index_loss"] = index_loss
            if lm_loss is not None:  # Only add if computed
                outputs_dict["lm_loss"] = lm_loss

        if not return_dict:
            # Ensure order matches expectations if return_dict is False
            # Be careful here, the order depends on what's calculated.
            # A safer approach might be to always return a dict or define a fixed tuple structure.
            output_values = [outputs_dict.get("token_logits"), outputs_dict.get("index_logits"),
                             outputs_dict.get("lm_logits")]
            if "loss" in outputs_dict:
                output_values.extend(
                    [outputs_dict.get("loss"), outputs_dict.get("token_loss"), outputs_dict.get("index_loss"),
                     outputs_dict.get("lm_loss")])
            # Filter out None values if necessary, depending on expected tuple structure
            return tuple(v for v in output_values if v is not None)

        return outputs_dict  # Use the renamed dictionary
