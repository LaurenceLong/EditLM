"""
编辑相关的工具函数
"""
from typing import List, Dict, Any, Tuple
import torch

def apply_edit(current_sequence: List[int], edit_index: int, token_id: int, tokenizer) -> List[int]:
    """
    应用编辑操作到当前序列

    Args:
        current_sequence: 当前文本序列的词元ID列表
        edit_index: 编辑操作的索引位置
        token_id: 要插入或删除的词元ID
        tokenizer: 词元器对象，用于获取特殊词元ID

    Returns:
        编辑后的序列
    """
    new_sequence = current_sequence.copy()

    # 如果是删除操作
    if token_id == tokenizer.del_token_id:
        if edit_index < len(new_sequence):
            new_sequence.pop(edit_index)
    # 如果是插入操作 (非EOS_EDIT)
    elif token_id != tokenizer.eos_token_id:
        new_sequence.insert(edit_index, token_id)

    return new_sequence

def predict_edit(model, tokenizer, input_sequence: List[int], device: str = "cuda") -> Tuple[int, int]:
    """
    使用模型预测单步编辑操作

    Args:
        model: 编辑模型
        tokenizer: 词元器
        input_sequence: 输入序列的词元ID列表
        device: 运行设备

    Returns:
        (预测的索引, 预测的词元ID)
    """
    model.eval()

    # 准备输入
    input_ids = torch.tensor([input_sequence], device=device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

    # 获取预测结果
    token_logits = outputs["token_logits"]
    index_logits = outputs["index_logits"]

    # 取最高概率的预测
    predicted_token_id = torch.argmax(token_logits, dim=-1).item()
    predicted_index = torch.argmax(index_logits, dim=-1).item()

    return predicted_index, predicted_token_id

def generate_edits(model, tokenizer, source_text: str, config, device: str = "cuda") -> str:
    """
    使用模型生成编辑序列，得到最终文本

    Args:
        model: 编辑模型
        tokenizer: 词元器
        source_text: 源文本
        config: 配置对象
        device: 运行设备

    Returns:
        编辑后的文本
    """
    # 词元化源文本
    source_ids = tokenizer.encode(source_text, add_special_tokens=False)
    current_sequence = source_ids.copy()

    for step in range(config.max_edit_steps):
        # 限制序列长度
        if len(current_sequence) > config.max_seq_length - 1:
            current_sequence = current_sequence[:config.max_seq_length - 1]

        # 预测编辑操作
        predicted_index, predicted_token_id = predict_edit(
            model, tokenizer, current_sequence, device
        )

        # 检查是否结束编辑
        if predicted_token_id == tokenizer.eos_token_id:
            break

        # 应用编辑
        current_sequence = apply_edit(
            current_sequence, predicted_index, predicted_token_id, tokenizer
        )

    # 将词元ID转换回文本
    edited_text = tokenizer.decode(current_sequence)

    return edited_text
