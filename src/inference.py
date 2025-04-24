import argparse
import json
import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

from utils import load_model_from_ckpt


def load_inputs(args):
    """根据不同的输入方式加载要处理的文本列表"""
    texts = []

    if args.input_text:
        texts = [args.input_text]

    elif args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]

    elif args.dataset:
        # 从HuggingFace加载数据集
        dataset = load_dataset(args.dataset, split=args.dataset_split)
        if args.max_samples:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))

        texts = dataset[args.text_column]

    elif args.dataset_path:
        # 加载自定义数据集文件
        import json
        with open(args.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            if all(isinstance(item, str) for item in data):
                texts = data
            elif all(isinstance(item, dict) for item in data):
                if args.text_column not in data[0]:
                    raise ValueError(f"Column '{args.text_column}' not found in dataset")
                texts = [item[args.text_column] for item in data]
        elif isinstance(data, dict) and args.text_column in data:
            texts = data[args.text_column]

    # 根据参数限制样本数量
    if args.max_samples and len(texts) > args.max_samples:
        texts = texts[:args.max_samples]

    return texts


def find_best_edit(model, tokenizer, text: str, max_len: Optional[int] = None) -> Dict:
    """
    找到文本最佳的编辑位置和编辑操作

    Args:
        model: EditLM模型
        tokenizer: 分词器
        text: 要编辑的文本
        max_len: 最大生成token数量限制

    Returns:
        包含原始文本、编辑后文本和编辑信息的字典
    """
    model.eval()
    device = next(model.parameters()).device

    # 对输入文本进行分词
    tokens = tokenizer.encode(text, add_special_tokens=False)
    input_ids = torch.tensor(tokens, device=device).unsqueeze(0)  # [1, seq_len]

    with torch.no_grad():
        # 获取模型预测
        outputs = model(input_ids=input_ids)

        # 获取预测的编辑位置和编辑token
        index_logits = outputs["index_logits"]  # [1, seq_len+1]
        token_logits = outputs["token_logits"]  # [1, vocab_size]
        pred_index = outputs["pred_index"].item()  # 标量

        # 获取token预测概率
        token_probs = F.softmax(token_logits, dim=-1)

        # 选择概率最高的token
        pred_token_id = token_probs.argmax(dim=-1).item()

    # 获取预测token的文本表示
    pred_token_text = tokenizer.decode([pred_token_id])

    # 根据预测的位置插入新token
    edited_tokens = tokens.copy()

    # 创建编辑信息字典
    edit_info = {
        "operation": "插入",
        "position": pred_index,
        "token": pred_token_text,
        "token_id": pred_token_id,
        "index_logits": index_logits[0].tolist()[:3],  # 仅保存前三个位置的logits用于调试
        "token_logits": token_logits[0].tolist()[:10]  # 仅保存前十个token的logits用于调试
    }

    # 插入预测的token
    edited_tokens.insert(pred_index, pred_token_id)

    # 如果指定了max_len，截断到指定长度
    if max_len is not None and len(edited_tokens) > max_len:
        edited_tokens = edited_tokens[:max_len]

    # 解码编辑后的tokens
    edited_text = tokenizer.decode(edited_tokens)

    return {
        "original_text": text,
        "edited_text": edited_text,
        "edit_info": edit_info
    }


def auto_edit_until_complete(model, tokenizer, text: str, max_len: Optional[int] = None, max_edits: int = 10) -> Dict:
    """
    自动编辑文本，直到文本完成或达到最大编辑次数

    Args:
        model: EditLM模型
        tokenizer: 分词器
        text: 要编辑的文本
        max_len: 最大token数量限制
        max_edits: 最大编辑次数，防止无限循环

    Returns:
        包含原始文本、最终编辑后文本和编辑历史的字典
    """
    original_text = text
    current_text = text
    edit_history = []

    # 计算初始token数量
    initial_tokens = tokenizer.encode(text, add_special_tokens=False)

    for edit_num in range(max_edits):
        # 进行一次编辑
        result = find_best_edit(model, tokenizer, current_text, max_len)

        # 更新当前文本
        current_text = result["edited_text"]

        # 记录此次编辑信息
        edit_history.append({
            "step": edit_num + 1,
            "edit": result["edit_info"],
            "result": current_text
        })

        # 检查是否达到停止条件

        # 1. 检查是否达到最大token长度
        if max_len is not None:
            current_tokens = tokenizer.encode(current_text, add_special_tokens=False)
            if len(current_tokens) >= max_len:
                break

        # 2. 检查是否生成结束（如生成了结束符号等，可根据具体模型特性调整）
        # 这里使用一个简单的启发式方法：检查最后是否是标点符号如句号、问号、感叹号等
        if current_text and current_text[-1] in ['.', '。', '?', '？', '!', '！']:
            if len(current_text) > len(original_text) + 10:  # 确保不是原始文本就带有的结束符
                break

    return {
        "original_text": original_text,
        "final_text": current_text,
        "edit_history": edit_history,
        "num_edits": len(edit_history)
    }


def batch_process(model, tokenizer, texts: List[str], args) -> List[Dict]:
    """批量处理多个文本"""
    results = []

    for text in tqdm(texts, desc="Processing texts"):
        if args.single_edit:
            # 单次编辑
            result = find_best_edit(model, tokenizer, text, args.max_len)
        else:
            # 连续编辑直到完成
            result = auto_edit_until_complete(
                model, tokenizer, text,
                max_len=args.max_len,
                max_edits=args.max_edits
            )

        results.append(result)

    return results


def save_results(results: List[Dict], args):
    """保存处理结果"""
    os.makedirs(args.output_dir, exist_ok=True)

    if len(results) == 1 and args.output_file is None:
        # 单个结果使用默认文件名
        output_file = os.path.join(args.output_dir, "single_result.json")
    else:
        # 多个结果或指定了输出文件
        output_file = args.output_file or os.path.join(args.output_dir, "batch_results.json")

    with open(output_file, 'w', encoding='utf-8') as fd:
        if len(results) == 1:
            # 单个结果直接保存
            json.dump(results[0], fd, ensure_ascii=False, indent=2)
        else:
            # 多个结果保存为列表
            json.dump(results, fd, ensure_ascii=False, indent=2)

    print(f"Results saved to {output_file}")

    # 同时输出文本摘要报告
    report_path = output_file.replace('.json', '_summary.txt')
    with open(report_path, 'w', encoding='utf-8') as fd:
        fd.write(f"编辑摘要报告 - 共处理{len(results)}个文本\n")
        fd.write("=" * 50 + "\n\n")

        for i, result in enumerate(results):
            fd.write(f"样本 #{i + 1}:\n")
            fd.write(f"原始文本: {result['original_text']}\n")

            if 'edited_text' in result:  # 单次编辑
                fd.write(f"编辑后文本: {result['edited_text']}\n")
                fd.write(
                    f"编辑操作: {result['edit_info']['operation']} {result['edit_info']['token']} at position {result['edit_info']['position']}\n")
            else:  # 连续编辑
                fd.write(f"最终文本: {result['final_text']}\n")
                fd.write(f"编辑次数: {result['num_edits']}\n")

            fd.write("\n" + "-" * 50 + "\n\n")

    print(f"Summary report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="使用EditLM进行文本编辑推理")

    # 模型参数
    parser.add_argument("--model_path", required=True, help="已训练的EditLM模型路径")
    parser.add_argument("--base_model", default=None, help="可选，指定基础模型")

    # 输入参数 (四选一)
    parser.add_argument("--input_text", help="单个输入文本")
    parser.add_argument("--input_file", help="输入文本文件路径，每行一个样本")
    parser.add_argument("--dataset", help="HuggingFace数据集名称")
    parser.add_argument("--dataset_path", help="自定义数据集文件路径")

    # 数据集相关参数
    parser.add_argument("--dataset_split", default="test", help="数据集分割")
    parser.add_argument("--text_column", default="text", help="文本所在的列名")
    parser.add_argument("--max_samples", type=int, help="最大处理样本数")

    # 推理参数
    parser.add_argument("--single_edit", action="store_true", default=False,
                        help="仅进行单次编辑而非连续编辑")
    parser.add_argument("--max_edits", type=int, default=20,
                        help="最大编辑次数限制 (连续编辑模式)")
    parser.add_argument("--max_len", type=int, default=None,
                        help="生成文本的最大token数限制")

    # 输出参数
    parser.add_argument("--output_dir", default="./inference_results",
                        help="输出目录")
    parser.add_argument("--output_file", help="输出文件名")

    args = parser.parse_args()

    # 检查输入参数
    input_methods = [args.input_text, args.input_file, args.dataset, args.dataset_path]
    if sum(m is not None for m in input_methods) != 1:
        parser.error("必须且只能指定一种输入方式: --input_text, --input_file, --dataset 或 --dataset_path")

    # 加载模型和tokenizer
    print(f"加载模型: {args.model_path}")
    model, tokenizer, _ = load_model_from_ckpt(args.model_path, base_model=args.base_model)

    # 将模型移至GPU (如果可用)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # 加载输入文本
    texts = load_inputs(args)
    print(f"加载了 {len(texts)} 个文本进行处理")

    # 批量处理文本
    results = batch_process(model, tokenizer, texts, args)

    # 保存结果
    save_results(results, args)

    # 打印示例结果
    if results:
        if 'edited_text' in results[0]:  # 单次编辑
            print(f"\n示例结果:")
            print(f"原始文本: {results[0]['original_text']}")
            print(f"编辑后: {results[0]['edited_text']}")
        else:  # 连续编辑
            print(f"\n示例结果:")
            print(f"原始文本: {results[0]['original_text']}")
            print(f"最终文本: {results[0]['final_text']}")
            print(f"编辑次数: {results[0]['num_edits']}")


if __name__ == "__main__":
    main()