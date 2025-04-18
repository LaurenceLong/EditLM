"""
数据预处理模块：将源文本-目标文本对转换为编辑序列
"""
from typing import List, Dict, Tuple, Any
import numpy as np
from edit_distance import SequenceMatcher

def compute_edit_operations(source_text: List[int], target_text: List[int], tokenizer) -> List[Dict[str, Any]]:
    """
    计算从源文本到目标文本的编辑操作序列

    Args:
        source_text: 源文本的词元ID序列
        target_text: 目标文本的词元ID序列
        tokenizer: 词元器对象，用于获取特殊词元ID

    Returns:
        编辑操作列表，每个操作包含 {'edit_index': int, 'token_id': int}
    """
    # 使用SequenceMatcher获取编辑操作
    matcher = SequenceMatcher(a=source_text, b=target_text)
    opcodes = matcher.get_opcodes()

    # 初始化当前序列为源文本
    current_sequence = source_text.copy()

    edit_operations = []
    offset = 0  # 操作后序列长度变化的累积偏移量

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            continue

        if tag == 'delete':
            # 处理删除操作
            for idx in range(i1, i2):
                adjusted_idx = idx + offset
                edit_operations.append({
                    'edit_index': adjusted_idx,
                    'token_id': tokenizer.del_token_id
                })
                # 更新当前序列
                current_sequence.pop(adjusted_idx)
                offset -= 1

        elif tag == 'insert':
            # 处理插入操作
            for idx, token_id in enumerate(target_text[j1:j2]):
                adjusted_idx = i1 + idx + offset
                edit_operations.append({
                    'edit_index': adjusted_idx,
                    'token_id': token_id
                })
                # 更新当前序列
                current_sequence.insert(adjusted_idx, token_id)
                offset += 1

        elif tag == 'replace':
            # 将替换拆解为删除+插入
            # 1. 先删除
            for idx in range(i1, i2):
                adjusted_idx = idx + offset
                edit_operations.append({
                    'edit_index': adjusted_idx,
                    'token_id': tokenizer.del_token_id
                })
                # 更新当前序列
                current_sequence.pop(adjusted_idx)
                offset -= 1

            # 2. 再插入
            for idx, token_id in enumerate(target_text[j1:j2]):
                adjusted_idx = i1 + idx + offset
                edit_operations.append({
                    'edit_index': adjusted_idx,
                    'token_id': token_id
                })
                # 更新当前序列
                current_sequence.insert(adjusted_idx, token_id)
                offset += 1

    # 添加最后一步的EOS_EDIT操作
    if edit_operations:
        last_edit_index = edit_operations[-1]['edit_index']
    else:
        last_edit_index = len(current_sequence)  # 如果没有编辑，则使用序列末尾

    edit_operations.append({
        'edit_index': last_edit_index,
        'token_id': tokenizer.eos_token_id
    })

    return edit_operations

def generate_training_samples(source_text: List[int], target_text: List[int], tokenizer) -> List[Dict[str, Any]]:
    """
    生成训练样本，每个样本包含当前状态、目标索引和目标词元

    Args:
        source_text: 源文本的词元ID序列
        target_text: 目标文本的词元ID序列
        tokenizer: 词元器对象

    Returns:
        训练样本列表
    """
    edit_operations = compute_edit_operations(source_text, target_text, tokenizer)

    training_samples = []
    current_sequence = source_text.copy()

    for op in edit_operations:
        # 记录当前状态作为输入
        training_samples.append({
            'input_state': current_sequence.copy(),
            'target_index': op['edit_index'],
            'target_token_id': op['token_id']
        })

        # 更新当前序列
        if op['token_id'] == tokenizer.del_token_id:
            if op['edit_index'] < len(current_sequence):
                current_sequence.pop(op['edit_index'])
        elif op['token_id'] != tokenizer.eos_token_id:
            current_sequence.insert(op['edit_index'], op['token_id'])

    return training_samples
