import argparse
import json
import os

import torch
from datasets import load_dataset

from tokenizer import get_del_token_id
from utils import load_model_from_ckpt


def load_model_for_inference(model_path, base_model="Qwen/Qwen2.5-0.5B", device="cuda"):
    """
    加载EditLMHF模型并准备进行推理
    """
    print(f"正在加载模型: {model_path}")
    model, tokenizer, _ = load_model_from_ckpt(model_path, base_model=base_model)
    model.to(device)
    model.eval()
    return model, tokenizer


def inference(model, tokenizer, text, device="cuda"):
    """
    使用模型进行推理，预测插入位置和应插入的token
    """
    # 将文本tokenize
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

    # 模型推理
    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    # 获取预测的插入位置
    index_logits = outputs["index_logits"]
    predicted_index = index_logits.argmax(-1).item()

    # 获取预测的token
    token_logits = outputs["token_logits"]
    predicted_token_id = token_logits.argmax(-1).item()
    predicted_token = tokenizer.decode([predicted_token_id])

    # 检查是否是删除token
    del_token_id = get_del_token_id(tokenizer)
    is_deletion = (predicted_token_id == del_token_id)

    # 执行编辑操作
    input_ids_list = input_ids[0].cpu().tolist()

    if is_deletion:
        # 删除操作
        if 0 <= predicted_index < len(input_ids_list):
            del input_ids_list[predicted_index]
        operation = "删除"
    else:
        # 插入操作
        if 0 <= predicted_index <= len(input_ids_list):
            input_ids_list.insert(predicted_index, predicted_token_id)
        operation = "插入"

    # 解码编辑后的文本
    result_text = tokenizer.decode(input_ids_list)

    # 返回结果
    edit_info = {
        "operation": operation,
        "position": predicted_index,
        "token": predicted_token,
        "token_id": predicted_token_id,
        "index_logits": index_logits[0].cpu().tolist(),  # 转换为列表而不是numpy数组，便于JSON序列化
        "token_logits": token_logits[0].cpu().tolist()[:10]  # 仅包含前10个logits以节省空间
    }

    return result_text, edit_info


def inference_batch(model, tokenizer, texts, device="cuda", output_file=None):
    """
    批量处理文本并保存结果
    """
    results = []

    for i, text in enumerate(texts):
        try:
            edited_text, edit_info = inference(model, tokenizer, text, device)

            result = {
                "id": i,
                "original_text": text,
                "edited_text": edited_text,
                "edit_info": edit_info
            }

            results.append(result)

            # 打印进度
            if (i + 1) % 10 == 0:
                print(f"已处理 {i + 1}/{len(texts)} 个样本")

        except Exception as e:
            print(f"处理样本 {i} 时出错: {e}")
            results.append({
                "id": i,
                "original_text": text,
                "error": str(e)
            })

    # 保存结果
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存至: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="使用EditLMHF模型进行文本编辑推理")
    parser.add_argument("--model_path", required=True, help="训练好的模型路径")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B", help="基础模型名称")
    parser.add_argument("--output_dir", default="./inference_results", help="输出目录")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="设备类型")

    # 输入方式选择
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_file", help="包含文本的输入文件，每行一个样本")
    input_group.add_argument("--input_text", help="单个输入文本")
    input_group.add_argument("--dataset", help="要使用的数据集名称，例如'gsm8k'")
    input_group.add_argument("--dataset_path", help="自定义数据集文件路径")

    # 数据集相关参数
    parser.add_argument("--dataset_config", help="数据集配置，例如'main'")
    parser.add_argument("--dataset_split", default="test", help="数据集划分，默认为'test'")
    parser.add_argument("--text_column", default="question", help="数据集中文本列的名称")
    parser.add_argument("--max_samples", type=int, help="处理的最大样本数量")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    model, tokenizer = load_model_for_inference(args.model_path, args.base_model, args.device)

    # 根据输入方式获取文本样本
    if args.input_text:
        # 单个文本
        texts = [args.input_text]
        output_file = os.path.join(args.output_dir, "single_result.json")
    elif args.input_file:
        # 从文件读取
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        output_file = os.path.join(args.output_dir, "file_results.json")
    elif args.dataset:
        # 从HuggingFace数据集加载
        config = args.dataset_config if args.dataset_config else None
        try:
            dataset = load_dataset(args.dataset, config, split=args.dataset_split)
            texts = dataset[args.text_column]
            if args.max_samples and args.max_samples < len(texts):
                texts = texts[:args.max_samples]
            output_file = os.path.join(args.output_dir, f"{args.dataset}_results.json")
        except Exception as e:
            print(f"加载数据集时出错: {e}")
            return
    elif args.dataset_path:
        # 从本地数据集文件加载
        try:
            if args.dataset_path.endswith('.json'):
                with open(args.dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    # 假设是对象列表
                    texts = [item.get(args.text_column, "") for item in data]
                else:
                    # 假设是字典
                    texts = data.get(args.text_column, [])
            elif args.dataset_path.endswith('.txt'):
                with open(args.dataset_path, 'r', encoding='utf-8') as f:
                    texts = [line.strip() for line in f if line.strip()]
            else:
                print(f"不支持的文件格式: {args.dataset_path}")
                return

            if args.max_samples and args.max_samples < len(texts):
                texts = texts[:args.max_samples]

            output_file = os.path.join(args.output_dir, "custom_dataset_results.json")
        except Exception as e:
            print(f"加载自定义数据集时出错: {e}")
            return

    # 执行推理
    print(f"开始处理 {len(texts)} 个文本样本...")

    if len(texts) == 1:
        # 单个文本样本
        edited_text, edit_info = inference(model, tokenizer, texts[0], args.device)

        # 打印结果
        print(f"\n原始文本: {texts[0]}")
        print(f"编辑操作: {edit_info['operation']}")
        print(f"编辑位置: {edit_info['position']}")
        print(f"编辑token: {edit_info['token']}")
        print(f"编辑后文本: {edited_text}")

        # 保存结果
        result = {
            "original_text": texts[0],
            "edited_text": edited_text,
            "edit_info": edit_info
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n结果已保存至: {output_file}")
    else:
        # 批量处理
        inference_batch(model, tokenizer, texts, args.device, output_file)


if __name__ == "__main__":
    main()