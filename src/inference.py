# src/inference.py
import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
# datasets >= 2.18 is recommended for better local file handling
from datasets import load_dataset
from tqdm import tqdm

from tokenizer import get_del_token_id  # Import the function to get delete token id
# Import necessary functions from project files
from utils import load_model_from_ckpt


# Helper function for sampling
def sample_top_k(logits: torch.Tensor, top_k: int, temperature: float = 1.0) -> torch.Tensor:
    """
    Sample from the logits using top-k sampling and temperature.
    Args:
        logits: Tensor of shape [batch_size, vocab_size]
        top_k: Keep only top_k tokens with highest probability.
        temperature: Softmax temperature for sampling.
    Returns:
        Sampled token indices of shape [batch_size, 1]
    """
    # Apply temperature
    logits = logits / temperature

    # Get top k logits and indices
    top_k = min(top_k, logits.size(-1))  # Ensure top_k is not larger than vocab size
    top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)  # [B, k]

    # Calculate probabilities using softmax on the top_k logits
    top_k_probs = F.softmax(top_k_logits, dim=-1)  # [B, k]

    # Sample from the top_k distribution
    sampled_relative_indices = torch.multinomial(top_k_probs, num_samples=1)  # [B, 1]

    # Map back to original vocabulary indices
    sampled_indices = torch.gather(top_k_indices, dim=-1, index=sampled_relative_indices)  # [B, 1]

    return sampled_indices


def load_inputs(args) -> Tuple[List[str], List[Optional[Dict]]]:
    """
    根据不同的输入方式加载要处理的文本列表和可选的元数据。

    Returns:
        Tuple[List[str], List[Optional[Dict]]]: A tuple containing the list of texts
                                                and a list of corresponding metadata (or None).
    """
    texts = []
    metadata = []  # Store corresponding metadata if available

    if args.input_text:
        texts = [args.input_text]
        metadata = [None] * len(texts)

    elif args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        metadata = [None] * len(texts)

    elif args.dataset or args.dataset_path:
        # Use datasets library for both HF datasets and local files
        load_path = args.dataset if args.dataset else args.dataset_path
        try:
            print(f"Loading dataset from: {load_path} using split: {args.dataset_split}")
            # Handle potential errors during loading (e.g., file not found, invalid format)
            dataset = load_dataset(load_path, split=args.dataset_split)

            # Ensure the text column exists
            if args.text_column not in dataset.column_names:
                raise ValueError(
                    f"Text column '{args.text_column}' not found in dataset columns: {dataset.column_names}"
                )

            # Limit samples if max_samples is set
            num_samples = len(dataset)
            if args.max_samples and args.max_samples < num_samples:
                dataset = dataset.select(range(args.max_samples))
                print(f"Limiting to {args.max_samples} samples.")

            # Extract texts and potentially other columns as metadata
            texts = dataset[args.text_column]
            # Store the rest of the row as metadata
            metadata_cols = {col: dataset[col] for col in dataset.column_names if col != args.text_column}
            # Convert columnar metadata to list of dicts
            num_rows = len(texts)
            metadata = [{col: metadata_cols[col][i] for col in metadata_cols} for i in range(num_rows)]

        except FileNotFoundError:
            print(f"Error: Dataset file not found at {load_path}")
            exit(1)
        except ValueError as e:
            print(f"Error loading or processing dataset: {e}")
            exit(1)
        except Exception as e:  # Catch other potential datasets library errors
            print(f"An unexpected error occurred while loading the dataset: {e}")
            exit(1)

    else:
        # This case should not be reached due to argument checking in main()
        print("Error: No valid input source specified.")
        exit(1)

    # Final check on sample limit (mainly for file inputs)
    if args.max_samples and len(texts) > args.max_samples:
        texts = texts[:args.max_samples]
        metadata = metadata[:args.max_samples]

    # Ensure metadata list has the same length as texts list
    if len(metadata) != len(texts):
        print(
            f"Warning: Metadata length ({len(metadata)}) mismatch with text length ({len(texts)}). Padding metadata with None.")
        metadata.extend([None] * (len(texts) - len(metadata)))

    return texts, metadata


def find_best_edit(model, tokenizer, text: str,
                   max_len: Optional[int] = None,
                   temperature: float = 1.0,
                   top_k: int = 50) -> Dict:
    """
    找到文本最佳的编辑位置和编辑操作 (插入、删除或追加).
    Handles sampling using temperature and top_k.

    Args:
        model: EditLM模型
        tokenizer: 分词器
        text: 要编辑的文本
        max_len: 最大生成token数量限制
        temperature: 采样温度
        top_k: Top-k 采样

    Returns:
        包含原始文本、编辑后文本和编辑信息的字典
    """
    model.eval()
    device = next(model.parameters()).device
    del_token_id = get_del_token_id(tokenizer)  # Get the delete token ID

    # 对输入文本进行分词
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:  # Handle empty input text
        return {
            "original_text": text,
            "edited_text": text,
            "edit_info": {"operation": "无操作", "reason": "输入为空"},
            "status": "skipped"
        }

    input_ids = torch.tensor(tokens, device=device).unsqueeze(0)  # [1, seq_len]
    current_seq_len = input_ids.shape[1]

    with torch.no_grad():
        # 获取模型预测
        outputs = model(input_ids=input_ids)  # Forward pass in inference mode

        # 获取预测的编辑位置和编辑token logits
        index_logits = outputs["index_logits"]  # [1, seq_len+1]
        token_logits = outputs["token_logits"]  # [1, vocab_size]

        # 1. 预测编辑位置 (使用 argmax for position)
        pred_index = index_logits.argmax(dim=-1).item()  # Scalar index [0, L]

        # 2. 预测编辑 Token (使用 top-k sampling)
        # Ensure token_logits is [1, vocab_size] before sampling
        if token_logits.ndim == 3:  # Should be [B, V] from model, but double check
            token_logits = token_logits.squeeze(0)  # Make it [V] if needed, or handle batch > 1
        if token_logits.ndim == 1:
            token_logits = token_logits.unsqueeze(0)  # Make it [1, V] for sampling function

        pred_token_id_tensor = sample_top_k(token_logits, top_k, temperature)  # [1, 1]
        pred_token_id = pred_token_id_tensor.item()  # Scalar token id

    # 获取预测token的文本表示 (if not deletion)
    pred_token_text = tokenizer.decode([pred_token_id]) if pred_token_id != del_token_id else "<删除>"

    # 根据预测的位置和 token 执行编辑
    edited_tokens = tokens.copy()
    edit_info = {
        "predicted_index": pred_index,
        "predicted_token_id": pred_token_id,
        "predicted_token_text": pred_token_text,
        # "index_logits_sample": index_logits[0].tolist()[:5], # Optional: Log some logits
        # "token_logits_sample": token_logits[0].tolist()[:10] # Optional: Log some logits
    }

    operation_performed = "无操作"
    details = ""

    # Case 1: Deletion
    if pred_token_id == del_token_id:
        if 0 <= pred_index < current_seq_len:
            deleted_token_id = edited_tokens.pop(pred_index)
            deleted_token_text = tokenizer.decode([deleted_token_id])
            operation_performed = "删除"
            details = f"在位置 {pred_index} 删除了 token '{deleted_token_text}' (ID: {deleted_token_id})"
        else:
            details = f"预测在位置 {pred_index} 删除，但该位置无效 (序列长度 {current_seq_len})。"
            operation_performed = "无效删除"

    # Case 2: Prediction / Append
    elif pred_index == current_seq_len:
        edited_tokens.append(pred_token_id)
        operation_performed = "追加"
        details = f"在末尾追加了 token '{pred_token_text}' (ID: {pred_token_id})"

    # Case 3: Insertion
    elif 0 <= pred_index < current_seq_len:
        edited_tokens.insert(pred_index, pred_token_id)
        operation_performed = "插入"
        details = f"在位置 {pred_index} 插入了 token '{pred_token_text}' (ID: {pred_token_id})"

    # Case 4: Invalid Index for Insertion/Append
    else:
        details = f"预测在位置 {pred_index} 插入/追加，但该位置无效 (序列长度 {current_seq_len})。"
        operation_performed = "无效插入/追加"

    # 更新编辑信息
    edit_info["operation"] = operation_performed
    edit_info["details"] = details

    # 如果指定了max_len，截断到指定长度
    final_edited_tokens = edited_tokens
    if max_len is not None and len(final_edited_tokens) > max_len:
        final_edited_tokens = final_edited_tokens[:max_len]
        edit_info["truncated_to_max_len"] = True

    # 解码编辑后的tokens
    edited_text = tokenizer.decode(final_edited_tokens)

    return {
        "original_text": text,
        "edited_text": edited_text,
        "edit_info": edit_info,
        "status": "edited" if operation_performed not in ["无操作", "无效删除", "无效插入/追加"] else "no_change"
    }


def auto_edit_until_complete(model, tokenizer, text: str,
                             max_len: Optional[int] = None,
                             max_edits: int = 10,
                             temperature: float = 1.0,
                             top_k: int = 50) -> Dict:
    """
    自动编辑文本，直到文本完成或达到最大编辑次数/长度。

    Args:
        model: EditLM模型
        tokenizer: 分词器
        text: 要编辑的文本
        max_len: 最大token数量限制
        max_edits: 最大编辑次数
        temperature: 采样温度
        top_k: Top-k 采样

    Returns:
        包含原始文本、最终编辑后文本和编辑历史的字典
    """
    original_text = text
    current_text = text
    edit_history = []
    last_text = text  # To detect no change

    for edit_num in range(max_edits):
        # 进行一次编辑
        result = find_best_edit(
            model, tokenizer, current_text,
            max_len=max_len,
            temperature=temperature,
            top_k=top_k
        )

        # 更新当前文本
        current_text = result["edited_text"]

        # 记录此次编辑信息
        edit_history.append({
            "step": edit_num + 1,
            "edit": result["edit_info"],
            "text_after_edit": current_text
        })

        # 检查是否达到停止条件

        # 1. 检查编辑是否有效或文本是否改变
        if result["status"] == "no_change" or current_text == last_text:
            print(f"停止编辑：在步骤 {edit_num + 1} 未发生有效改变。")
            break

        last_text = current_text  # Update last_text for next iteration check

        # 2. 检查是否达到最大token长度 (find_best_edit already handles truncation)
        current_tokens = tokenizer.encode(current_text, add_special_tokens=False)
        if max_len is not None and len(current_tokens) >= max_len:
            print(f"停止编辑：达到最大长度 {max_len}。")
            break

        # 3. 检查是否生成结束 (Optional: check for EOS token if applicable)
        # eos_token_id = tokenizer.eos_token_id
        # if eos_token_id and current_tokens and current_tokens[-1] == eos_token_id:
        #     print(f"停止编辑：生成了 EOS token。")
        #     break

        # Simple heuristic (keep or remove based on need)
        # if current_text and current_text[-1] in ['.', '。', '?', '？', '!', '！']:
        #     if len(current_text) > len(original_text) + 5: # Ensure some generation happened
        #         print(f"停止编辑：检测到结尾标点符号。")
        #         break

    return {
        "original_text": original_text,
        "final_text": current_text,
        "edit_history": edit_history,
        "num_edits": len(edit_history)
    }


def batch_process(model, tokenizer, texts: List[str], metadata: List[Optional[Dict]], args) -> List[Dict]:
    """
    批量处理多个文本。

    Args:
        model: The EditLM model.
        tokenizer: The tokenizer.
        texts: List of input texts.
        metadata: List of corresponding metadata dictionaries (or None).
        args: Command line arguments.

    Returns:
        List of result dictionaries.
    """
    results = []
    if len(texts) != len(metadata):
        raise ValueError("Texts and metadata lists must have the same length.")

    for i, text in enumerate(tqdm(texts, desc="Processing texts")):
        meta = metadata[i]  # Get metadata for this text
        if args.single_edit:
            # 单次编辑
            result = find_best_edit(
                model, tokenizer, text,
                max_len=args.max_len,
                temperature=args.temperature,
                top_k=args.top_k
            )
        else:
            # 连续编辑直到完成
            result = auto_edit_until_complete(
                model, tokenizer, text,
                max_len=args.max_len,
                max_edits=args.max_edits,
                temperature=args.temperature,
                top_k=args.top_k
            )

        # Add metadata to the result if it exists
        if meta:
            result["metadata"] = meta
        results.append(result)

    return results


def save_results(results: List[Dict], args):
    """保存处理结果到JSON和文本摘要文件"""
    # Determine output directory and file
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Determine the base filename
    if args.output_file:
        # Use user-provided filename (potentially with path)
        output_file_path = args.output_file
        # Ensure the directory exists if a path is included in output_file
        output_file_dir = os.path.dirname(output_file_path)
        if output_file_dir and not os.path.exists(output_file_dir):
            os.makedirs(output_file_dir)
    elif len(results) == 1 and not args.dataset and not args.dataset_path and not args.input_file:
        # Single input text, default filename
        output_file_path = os.path.join(output_dir, "single_result.json")
    else:
        # Batch processing or dataset input, default filename
        output_file_path = os.path.join(output_dir, "batch_results.json")

    # Save main results to JSON
    try:
        with open(output_file_path, 'w', encoding='utf-8') as fd:
            # Save as a list even if there's only one result for consistency
            json.dump(results, fd, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_file_path}")
    except Exception as e:
        print(f"Error saving JSON results to {output_file_path}: {e}")
        return  # Stop if saving fails

    # Save summary report to TXT
    report_path = os.path.splitext(output_file_path)[0] + '_summary.txt'
    try:
        with open(report_path, 'w', encoding='utf-8') as fd:
            fd.write(f"编辑摘要报告 - 共处理 {len(results)} 个文本\n")
            fd.write(f"模型: {args.model_path}\n")
            fd.write(f"模式: {'单次编辑' if args.single_edit else '连续编辑'}\n")
            if not args.single_edit:
                fd.write(f"最大编辑次数: {args.max_edits}\n")
            fd.write(f"最大长度: {args.max_len if args.max_len else '无限制'}\n")
            fd.write(f"采样温度: {args.temperature}\n")
            fd.write(f"Top-k: {args.top_k}\n")
            fd.write("=" * 50 + "\n\n")

            for i, result in enumerate(results):
                fd.write(f"样本 #{i + 1}:\n")
                if "metadata" in result and result["metadata"]:
                    fd.write(f"元数据: {json.dumps(result['metadata'], ensure_ascii=False)}\n")
                fd.write(f"原始文本: {result['original_text']}\n")

                if 'edited_text' in result:  # 单次编辑结果格式
                    fd.write(f"编辑后文本: {result['edited_text']}\n")
                    if "edit_info" in result:
                        fd.write(f"编辑操作: {result['edit_info'].get('operation', 'N/A')}\n")
                        fd.write(f"详细信息: {result['edit_info'].get('details', 'N/A')}\n")
                elif 'final_text' in result:  # 连续编辑结果格式
                    fd.write(f"最终文本: {result['final_text']}\n")
                    fd.write(f"编辑次数: {result['num_edits']}\n")
                    # Optionally include summary of edits if needed
                    # for edit_step in result.get('edit_history', []):
                    #     fd.write(f"  - 步骤 {edit_step['step']}: {edit_step['edit'].get('details', 'N/A')}\n")

                fd.write("\n" + "-" * 50 + "\n\n")
        print(f"Summary report saved to {report_path}")
    except Exception as e:
        print(f"Error saving summary report to {report_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="使用EditLM进行文本编辑推理")

    # --- Model Args ---
    parser.add_argument("--model_path", required=True, help="已训练的EditLM模型检查点路径 (.pt)")
    parser.add_argument("--base_model", default=None,
                        help="可选，指定基础模型名称 (例如 'Qwen/Qwen2.5-0.5B')。如果检查点目录中找不到tokenizer，则需要此项。")

    # --- Input Args (Choose One) ---
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_text", help="直接提供单个输入文本")
    input_group.add_argument("--input_file", help="提供包含输入文本的文件路径 (每行一个样本)")
    input_group.add_argument("--dataset", help="HuggingFace数据集名称 (例如 'wikitext', 'gsm8k')")
    input_group.add_argument("--dataset_path", help="本地数据集文件路径 (支持 json, csv, parquet, txt等)")

    # --- Dataset Specific Args (Used with --dataset or --dataset_path) ---
    parser.add_argument("--dataset_split", default="test",
                        help="要使用的数据集划分 (例如 'train', 'validation', 'test')")
    parser.add_argument("--text_column", default="text", help="数据集中包含文本内容的列名")
    parser.add_argument("--max_samples", type=int, default=None, help="限制处理的最大样本数量")

    # --- Inference Control Args ---
    parser.add_argument("--single_edit", action="store_true", default=False,
                        help="执行单次编辑模式，而不是连续编辑/生成模式")
    parser.add_argument("--max_edits", type=int, default=20,
                        help="最大编辑/生成步数 (仅用于连续编辑模式)")
    parser.add_argument("--max_len", type=int, default=None,
                        help="编辑或生成后的最大序列长度 (token数)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="采样温度，较低的值使输出更确定性 (例如 0.7)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k 采样，只考虑概率最高的 k 个词 (例如 50)")

    # --- Output Args ---
    parser.add_argument("--output_dir", default="./inference_results",
                        help="保存结果的目录")
    parser.add_argument("--output_file", default=None,
                        help="指定输出JSON文件的名称 (可选，默认为 'single_result.json' 或 'batch_results.json')")

    args = parser.parse_args()

    # Input validation (already handled by mutually_exclusive_group)

    # 加载模型和tokenizer
    print(f"Loading model and tokenizer from: {args.model_path}")
    try:
        # Pass base_model in case tokenizer isn't saved with the checkpoint
        model, tokenizer, _ = load_model_from_ckpt(args.model_path, base_model=args.base_model)
    except Exception as e:
        print(f"Error loading model from checkpoint: {e}")
        print(
            "Please ensure the model path is correct and --base_model is provided if the tokenizer isn't in the checkpoint directory.")
        exit(1)

    # 将模型移至GPU (如果可用)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = model.to(device)
        model.eval()
        print(f"Model loaded successfully on device: {device}")
    except Exception as e:
        print(f"Error moving model to device {device}: {e}")
        exit(1)

    # 加载输入文本和元数据
    print("Loading input data...")
    texts, metadata = load_inputs(args)
    if not texts:
        print("Error: No input texts were loaded. Please check your input arguments.")
        exit(1)
    print(f"Loaded {len(texts)} text sample(s) for processing.")

    # 批量处理文本
    print("Starting batch processing...")
    results = batch_process(model, tokenizer, texts, metadata, args)  # Pass tokenizer and args

    # 保存结果
    print("Saving results...")
    save_results(results, args)

    # 打印第一个结果的示例
    if results:
        print("\n--- Example Result (First Sample) ---")
        first_result = results[0]
        print(f"Original Text: {first_result.get('original_text', 'N/A')}")
        if args.single_edit:
            print(f"Edited Text: {first_result.get('edited_text', 'N/A')}")
            edit_info = first_result.get('edit_info', {})
            print(f"Operation: {edit_info.get('operation', 'N/A')}")
            print(f"Details: {edit_info.get('details', 'N/A')}")
        else:
            print(f"Final Text: {first_result.get('final_text', 'N/A')}")
            print(f"Number of Edits: {first_result.get('num_edits', 'N/A')}")
        print("--- End Example ---")

    print("\nInference finished.")


if __name__ == "__main__":
    main()
