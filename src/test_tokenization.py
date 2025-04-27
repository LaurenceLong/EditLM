"""
测试脚本：验证预处理后的数据集，并展示如何从每种任务中恢复原始数据
"""
import argparse
import os
import traceback

import numpy as np
import torch
from termcolor import colored

from tokenizer import get_tokenizer, get_del_token_id
from data import DeletionTaskDataset, InsertionTaskDataset, PredictionTaskDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="预处理数据目录")
    parser.add_argument("--base", default="Qwen/Qwen2.5-0.5B", help="模型名称，用于加载tokenizer")
    parser.add_argument("--num_samples", type=int, default=5, help="每种任务要展示的样本数量")
    args = parser.parse_args()

    # 加载tokenizer，用于将token ID转换回文本
    print(f"加载tokenizer: \n{args.base}")
    tokenizer = get_tokenizer(args.base)
    del_token_id = get_del_token_id(tokenizer)

    # 加载元数据
    metadata_path = os.path.join(args.data_dir, "train_metadata.pt")
    metadata = torch.load(metadata_path)
    seq_len = metadata["seq_len"]

    print("\n" + "="*80)
    print(f"数据目录: \n{args.data_dir}")
    print(f"序列长度: \n{seq_len}")
    print("="*80 + "\n")

    # 测试任务1：删除任务
    print("\n=== 测试任务1：删除任务 ===\n")
    try:
        # 创建删除任务数据集
        deletion_dir = os.path.join(args.data_dir, "deletion")
        deletion_dataset = DeletionTaskDataset(deletion_dir, "train", shuffle=False, del_token_id=del_token_id)
        print(f"成功创建删除任务数据集")

        # 加载几个样本
        for i in range(min(args.num_samples, len(deletion_dataset))):
            sample = deletion_dataset[i]

            sequence = sample['sequence']
            target_index = sample['index'].item()

            # 显示为文本
            text = tokenizer.decode(sequence)
            token_to_delete = sequence[target_index].item()
            token_to_delete_text = tokenizer.decode([token_to_delete])

            print(f"\n样本 {i+1}:")
            print(f"输入序列: \n{text[:1024]}..." if len(text) > 1024 else f"输入序列: \n{text}")
            print(f"删除目标: 位置 = {target_index}, Token = {token_to_delete} ('{token_to_delete_text}')")

            # 验证：执行删除操作，恢复原始序列
            print("验证 - 任务：从序列中删除指定位置的token")

            # 创建带颜色标记的序列，突出显示要删除的token
            tokens_list = sequence.tolist()
            colored_text = ""
            for j, token in enumerate(tokens_list):
                if j == target_index:
                    # 用红色标记要删除的token
                    token_text = tokenizer.decode([token])
                    colored_text += f"[RED]{token_text}[/RED]"
                else:
                    token_text = tokenizer.decode([token])
                    colored_text += token_text

            # 执行删除操作
            modified_tokens = tokens_list.copy()
            modified_tokens.pop(target_index)
            # 为了保持长度一致，在末尾添加一个填充token
            modified_tokens.append(tokenizer.pad_token_id)
            modified_text = tokenizer.decode(modified_tokens)

            print(f"输入序列（标记删除位置）: \n{colored_text[:1024].replace('[RED]', colored('', 'red', attrs=['bold'])).replace('[/RED]', '')}..." if len(colored_text) > 1024 else f"输入序列（标记删除位置）: \n{colored_text.replace('[RED]', colored('', 'red', attrs=['bold'])).replace('[/RED]', '')}")
            print(f"执行删除后的序列: \n{modified_text[:1024]}..." if len(modified_text) > 1024 else f"执行删除后的序列: \n{modified_text}")
    except Exception as e:
        print(f"测试删除任务时发生错误: \n{e}")
        traceback.print_exc()

    # 测试任务2：插入任务
    print("\n=== 测试任务2：插入任务 ===\n")
    try:
        # 创建插入任务数据集
        insertion_dir = os.path.join(args.data_dir, "insertion")
        insertion_dataset = InsertionTaskDataset(insertion_dir, "train", shuffle=False)
        print(f"成功创建插入任务数据集")

        # 加载几个样本
        for i in range(min(args.num_samples, len(insertion_dataset))):
            sample = insertion_dataset[i]

            sequence = sample['sequence']
            target_index = sample['index'].item()
            target_token = sample['token'].item()

            # 显示为文本
            text = tokenizer.decode(sequence)
            target_token_text = tokenizer.decode([target_token])

            print(f"\n样本 {i+1}:")
            print(f"输入序列: \n{text[:1024]}..." if len(text) > 1024 else f"输入序列: \n{text}")
            print(f"插入目标: 位置 = {target_index}, Token = {target_token} ('{target_token_text}')")

            # 验证：执行插入操作，恢复原始序列
            print("验证 - 任务：在序列中的指定位置插入特定token")

            # 创建带颜色标记的序列，突出显示要插入的位置
            tokens_list = sequence.tolist()
            colored_text = ""
            for j, token in enumerate(tokens_list):
                if j == target_index:
                    # 在插入位置标记
                    token_text = tokenizer.decode([token])
                    colored_text += f"[GREEN]^[/GREEN]{token_text}"
                else:
                    token_text = tokenizer.decode([token])
                    colored_text += token_text

            # 执行插入操作
            modified_tokens = tokens_list.copy()
            modified_tokens.insert(target_index, target_token)
            # 为了保持长度一致，删除最后一个token（通常是填充token）
            modified_tokens = modified_tokens[:-1]
            modified_text = tokenizer.decode(modified_tokens)

            print(f"输入序列（标记插入位置）: \n{colored_text[:1024].replace('[GREEN]', colored('', 'green', attrs=['bold'])).replace('[/GREEN]', '')}..." if len(colored_text) > 1024 else f"输入序列（标记插入位置）: \n{colored_text.replace('[GREEN]', colored('', 'green', attrs=['bold'])).replace('[/GREEN]', '')}")
            print(f"执行插入后的序列: \n{modified_text[:1024]}..." if len(modified_text) > 1024 else f"执行插入后的序列: \n{modified_text}")
            print(f"插入的token: '{target_token_text}'")
    except Exception as e:
        print(f"测试插入任务时发生错误: \n{e}")
        traceback.print_exc()

        # 测试任务3：预测任务
        print("\n=== 测试任务3：预测任务 ===\n")
        try:
            # 创建预测任务数据集
            prediction_dir = os.path.join(args.data_dir, "prediction")
            prediction_dataset = PredictionTaskDataset(prediction_dir, "train", shuffle=False)
            print(f"成功创建预测任务数据集")

            # 加载几个样本
            for i in range(min(args.num_samples, len(prediction_dataset))):
                sample = prediction_dataset[i]

                sequence = sample['sequence']
                target_index = sample['index'].item()
                target_token = sample['token'].item()

                # 显示为文本
                text = tokenizer.decode(sequence)
                target_token_text = tokenizer.decode([target_token])

                print(f"\n样本 {i + 1}:")
                print(f"输入序列: \n{text[:1024]}..." if len(text) > 1024 else f"输入序列: \n{text}")
                print(f"预测目标: 位置 = {target_index}, Token = {target_token} ('{target_token_text}')")

                # 验证：执行预测操作，恢复原始序列
                print("验证 - 任务：在序列中的指定位置预测特定token")

                # 创建带颜色标记的序列，突出显示要预测的位置
                tokens_list = sequence.tolist()
                colored_text = ""
                for j, token in enumerate(tokens_list):
                    if j == target_index:
                        # 在预测位置标记
                        token_text = tokenizer.decode([token])
                        colored_text += f"[GREEN]^[/GREEN]{token_text}"
                    else:
                        token_text = tokenizer.decode([token])
                        colored_text += token_text

                # 执行预测操作
                modified_tokens = tokens_list.copy()
                modified_tokens.insert(target_index, target_token)
                # 为了保持长度一致，删除最后一个token（通常是填充token）
                modified_tokens = modified_tokens[:-1]
                modified_text = tokenizer.decode(modified_tokens)

                print(
                    f"输入序列（标记预测位置）: \n{colored_text[:1024].replace('[GREEN]', colored('', 'green', attrs=['bold'])).replace('[/GREEN]', '')}..." if len(
                        colored_text) > 1024 else f"输入序列（标记预测位置）: \n{colored_text.replace('[GREEN]', colored('', 'green', attrs=['bold'])).replace('[/GREEN]', '')}")
                print(f"执行预测后的序列: \n{modified_text[:1024]}..." if len(
                    modified_text) > 1024 else f"执行预测后的序列: \n{modified_text}")
                print(f"预测的token: '{target_token_text}'")
        except Exception as e:
            print(f"测试预测任务时发生错误: \n{e}")
            traceback.print_exc()

    print("\n=== 测试完成 ===\n")


if __name__ == "__main__":
    main()
