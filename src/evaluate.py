import argparse
import json
import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import DeletionTaskDataset, InsertionTaskDataset, collate_fn
from model_hf import EditLMHF
from tokenizer import get_tokenizer, get_del_token_id
from utils import load_model_from_ckpt


class ModelEvaluator:
    def __init__(self, base_model="Qwen/Qwen2.5-0.5B", trained_model_path=None, device="cuda"):
        """
        初始化评估器，加载原始模型和训练后的模型
        """
        self.device = device
        self.base_model_name = base_model

        # 加载tokenizer
        self.tokenizer = get_tokenizer(base_model, use_fast=True)
        self.del_token_id = get_del_token_id(self.tokenizer)

        # 加载原始模型
        print(f"加载原始模型: {base_model}")
        self.base_model = EditLMHF(base_model=base_model)  # 创建未训练的EditLMHF
        self.base_model.to(device)
        self.base_model.eval()

        # 加载训练后的模型（如果有）
        self.trained_model = None
        if trained_model_path and os.path.exists(trained_model_path):
            print(f"加载训练后的模型: {trained_model_path}")
            self.trained_model, _, _ = load_model_from_ckpt(
                trained_model_path, base_model=base_model
            )
            self.trained_model.to(device)
            self.trained_model.eval()

    def evaluate_on_dataset(self, data_dir, task_type="insertion", split="validation", batch_size=8, num_samples=None):
        """
        在数据集上评估模型性能
        """
        # 加载适当的数据集
        if task_type == "deletion":
            dataset_dir = os.path.join(data_dir, "deletion")
            dataset = DeletionTaskDataset(dataset_dir, split, shuffle=False, del_token_id=self.del_token_id)
        else:  # insertion
            dataset_dir = os.path.join(data_dir, "insertion")
            dataset = InsertionTaskDataset(dataset_dir, split, shuffle=False)

        # 限制样本数
        if num_samples is not None:
            dataset = torch.utils.data.Subset(dataset, list(range(min(num_samples, len(dataset)))))

        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        # 初始化结果
        results = {
            "base_model": {
                "position_accuracy": 0,
                "token_accuracy": 0,
                "combined_accuracy": 0,
            },
            "trained_model": {
                "position_accuracy": 0,
                "token_accuracy": 0,
                "combined_accuracy": 0,
            } if self.trained_model else None,
            "samples_evaluated": 0
        }

        # 遍历数据集
        total_samples = 0
        base_pos_correct = 0
        base_token_correct = 0
        base_combined_correct = 0

        trained_pos_correct = 0
        trained_token_correct = 0
        trained_combined_correct = 0

        for batch in tqdm(dataloader, desc=f"评估{task_type}任务"):
            # 获取批次数据
            sequences = batch["sequences"].to(self.device)
            target_indices = batch["indices"].to(self.device)
            target_tokens = batch["tokens"].to(self.device)

            batch_size = sequences.size(0)
            total_samples += batch_size

            # 使用基础模型评估
            with torch.no_grad():
                base_outputs = self.base_model(input_ids=sequences)

            base_pred_indices = base_outputs["index_logits"].argmax(dim=-1)
            base_pred_tokens = base_outputs["token_logits"].argmax(dim=-1)

            # 计算准确率
            base_pos_correct += (base_pred_indices == target_indices).sum().item()
            base_token_correct += (base_pred_tokens == target_tokens).sum().item()
            base_combined_correct += ((base_pred_indices == target_indices) &
                                      (base_pred_tokens == target_tokens)).sum().item()

            # 如果有训练后的模型，也评估它
            if self.trained_model:
                with torch.no_grad():
                    trained_outputs = self.trained_model(input_ids=sequences)

                trained_pred_indices = trained_outputs["index_logits"].argmax(dim=-1)
                trained_pred_tokens = trained_outputs["token_logits"].argmax(dim=-1)

                # 计算准确率
                trained_pos_correct += (trained_pred_indices == target_indices).sum().item()
                trained_token_correct += (trained_pred_tokens == target_tokens).sum().item()
                trained_combined_correct += ((trained_pred_indices == target_indices) &
                                             (trained_pred_tokens == target_tokens)).sum().item()

        # 计算最终准确率
        results["base_model"]["position_accuracy"] = base_pos_correct / total_samples
        results["base_model"]["token_accuracy"] = base_token_correct / total_samples
        results["base_model"]["combined_accuracy"] = base_combined_correct / total_samples

        if self.trained_model:
            results["trained_model"]["position_accuracy"] = trained_pos_correct / total_samples
            results["trained_model"]["token_accuracy"] = trained_token_correct / total_samples
            results["trained_model"]["combined_accuracy"] = trained_combined_correct / total_samples

        results["samples_evaluated"] = total_samples

        return results

    def evaluate_text_correction(self, texts, correction_type="repetition", verbose=False):
        """
        评估文本纠错能力

        Args:
            texts: 文本列表
            correction_type: 纠错类型 ("repetition", "typo", "grammar")
            verbose: 是否打印详细信息

        Returns:
            results: 包含评估结果的字典
        """
        results = {
            "base_model": {
                "correction_rate": 0,
                "examples": []
            },
            "trained_model": {
                "correction_rate": 0,
                "examples": []
            } if self.trained_model else None,
            "samples_evaluated": len(texts)
        }

        base_corrected = 0
        trained_corrected = 0

        for i, text in enumerate(tqdm(texts, desc=f"评估{correction_type}纠错")):
            # 基础模型推理
            input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                base_outputs = self.base_model(input_ids=input_ids)

            base_pred_index = base_outputs["index_logits"].argmax(-1).item()
            base_pred_token_id = base_outputs["token_logits"].argmax(-1).item()

            # 执行基础模型的编辑
            base_edit_ids = input_ids[0].cpu().tolist()

            if base_pred_token_id == self.del_token_id:
                # 删除操作
                if 0 <= base_pred_index < len(base_edit_ids):
                    base_edit_ids.pop(base_pred_index)
            else:
                # 插入操作
                if 0 <= base_pred_index <= len(base_edit_ids):
                    base_edit_ids.insert(base_pred_index, base_pred_token_id)

            base_edited_text = self.tokenizer.decode(base_edit_ids)

            # 检查是否纠正了错误
            base_corrected_error = False
            if correction_type == "repetition":
                # 例如 "错误错误" -> "错误"
                base_corrected_error = len(base_edited_text) < len(text) and not _contains_repetition(base_edited_text)
            elif correction_type == "typo":
                # 简单的错别字检查
                base_corrected_error = base_edited_text != text
            elif correction_type == "grammar":
                # 语法错误更难自动检测，仅检查文本是否变化
                base_corrected_error = base_edited_text != text

            if base_corrected_error:
                base_corrected += 1

            if verbose or i < 5:  # 始终打印前5个示例
                results["base_model"]["examples"].append({
                    "original": text,
                    "edited": base_edited_text,
                    "corrected": base_corrected_error
                })

            # 训练后模型评估
            if self.trained_model:
                with torch.no_grad():
                    trained_outputs = self.trained_model(input_ids=input_ids)

                trained_pred_index = trained_outputs["index_logits"].argmax(-1).item()
                trained_pred_token_id = trained_outputs["token_logits"].argmax(-1).item()

                # 执行训练后模型的编辑
                trained_edit_ids = input_ids[0].cpu().tolist()

                if trained_pred_token_id == self.del_token_id:
                    # 删除操作
                    if 0 <= trained_pred_index < len(trained_edit_ids):
                        trained_edit_ids.pop(trained_pred_index)
                else:
                    # 插入操作
                    if 0 <= trained_pred_index <= len(trained_edit_ids):
                        trained_edit_ids.insert(trained_pred_index, trained_pred_token_id)

                trained_edited_text = self.tokenizer.decode(trained_edit_ids)

                # 检查是否纠正了错误
                trained_corrected_error = False
                if correction_type == "repetition":
                    trained_corrected_error = len(trained_edited_text) < len(text) and not _contains_repetition(
                        trained_edited_text)
                elif correction_type == "typo":
                    trained_corrected_error = trained_edited_text != text
                elif correction_type == "grammar":
                    trained_corrected_error = trained_edited_text != text

                if trained_corrected_error:
                    trained_corrected += 1

                if verbose or i < 5:  # 始终打印前5个示例
                    results["trained_model"]["examples"].append({
                        "original": text,
                        "edited": trained_edited_text,
                        "corrected": trained_corrected_error
                    })

        # 计算纠错率
        results["base_model"]["correction_rate"] = base_corrected / len(texts)
        if self.trained_model:
            results["trained_model"]["correction_rate"] = trained_corrected / len(texts)

        return results

    def evaluate_on_math_dataset(self, dataset_name="gsm8k", split="test", num_samples=None):
        """
        在数学数据集上评估编辑能力

        Args:
            dataset_name: 数据集名称
            split: 数据集划分
            num_samples: 样本数量限制

        Returns:
            results: 包含评估结果的字典
        """
        print(f"加载{dataset_name}数据集...")

        try:
            dataset = load_dataset(dataset_name, split=split)
        except Exception as e:
            print(f"加载数据集出错: {e}")
            return None

        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        # 获取问题文本
        if dataset_name == "gsm8k":
            texts = dataset["question"]
        elif dataset_name == "math_qa":
            texts = dataset["Problem"]
        else:
            # 尝试常见的列名
            if "question" in dataset.column_names:
                texts = dataset["question"]
            elif "problem" in dataset.column_names:
                texts = dataset["problem"]
            else:
                print(f"未知的数据集格式，请指定文本列名")
                return None

        # 创建包含加法错误的样本
        math_texts = []
        for text in texts:
            # 查找所有数字并随机引入加法错误
            modified_text = introduce_math_errors(text)
            if modified_text != text:  # 只保留成功引入错误的样本
                math_texts.append(modified_text)

        print(f"生成了 {len(math_texts)} 个包含数学错误的样本")

        # 评估模型进行纠错的能力
        correction_results = self.evaluate_text_correction(
            math_texts, correction_type="math", verbose=True
        )

        return correction_results

    def evaluate_qualitative(self, texts, verbose=True):
        """
        对一组文本样本进行质性评估
        """
        results = []

        for text in texts:
            # 将文本tokenize
            input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)

            # 基础模型推理
            with torch.no_grad():
                base_outputs = self.base_model(input_ids=input_ids)

            base_pred_index = base_outputs["index_logits"].argmax(-1).item()
            base_pred_token_id = base_outputs["token_logits"].argmax(-1).item()
            base_pred_token = self.tokenizer.decode([base_pred_token_id])

            # 执行基础模型的编辑
            base_edit_ids = input_ids[0].cpu().tolist()

            if base_pred_token_id == self.del_token_id:
                # 删除操作
                if 0 <= base_pred_index < len(base_edit_ids):
                    base_edit_ids.pop(base_pred_index)
                base_operation = "删除"
            else:
                # 插入操作
                if 0 <= base_pred_index <= len(base_edit_ids):
                    base_edit_ids.insert(base_pred_index, base_pred_token_id)
                base_operation = "插入"

            base_edited_text = self.tokenizer.decode(base_edit_ids)

            # 训练后模型推理（如果有）
            trained_results = None
            if self.trained_model:
                with torch.no_grad():
                    trained_outputs = self.trained_model(input_ids=input_ids)

                trained_pred_index = trained_outputs["index_logits"].argmax(-1).item()
                trained_pred_token_id = trained_outputs["token_logits"].argmax(-1).item()
                trained_pred_token = self.tokenizer.decode([trained_pred_token_id])

                # 执行训练后模型的编辑
                trained_edit_ids = input_ids[0].cpu().tolist()

                if trained_pred_token_id == self.del_token_id:
                    # 删除操作
                    if 0 <= trained_pred_index < len(trained_edit_ids):
                        trained_edit_ids.pop(trained_pred_index)
                    trained_operation = "删除"
                else:
                    # 插入操作
                    if 0 <= trained_pred_index <= len(trained_edit_ids):
                        trained_edit_ids.insert(trained_pred_index, trained_pred_token_id)
                    trained_operation = "插入"

                trained_edited_text = self.tokenizer.decode(trained_edit_ids)

                trained_results = {
                    "predicted_index": trained_pred_index,
                    "predicted_token": trained_pred_token,
                    "operation": trained_operation,
                    "edited_text": trained_edited_text
                }

            # 收集结果
            sample_result = {
                "original_text": text,
                "base_model": {
                    "predicted_index": base_pred_index,
                    "predicted_token": base_pred_token,
                    "operation": base_operation,
                    "edited_text": base_edited_text
                },
                "trained_model": trained_results
            }

            results.append(sample_result)

            # 打印结果
            if verbose:
                print("\n" + "-" * 80)
                print(f"原始文本: {text}")
                print("\n基础模型:")
                print(f"  编辑操作: {base_operation}")
                print(f"  编辑位置: {base_pred_index}")
                print(f"  编辑token: {base_pred_token}")
                print(f"  编辑后文本: {base_edited_text}")

                if self.trained_model:
                    print("\n训练后模型:")
                    print(f"  编辑操作: {trained_operation}")
                    print(f"  编辑位置: {trained_pred_index}")
                    print(f"  编辑token: {trained_pred_token}")
                    print(f"  编辑后文本: {trained_edited_text}")

        return results

    def run_comprehensive_evaluation(self, data_dir, output_dir, qualitative_texts=None, num_dataset_samples=500):
        """
        运行综合评估并保存结果
        """
        os.makedirs(output_dir, exist_ok=True)

        comprehensive_results = {
            "base_model": self.base_model_name,
            "trained_model": "None" if self.trained_model is None else "Trained EditLM",
            "quantitative": {},
            "qualitative": None
        }

        # 量化评估 - 插入任务
        print("\n\n===== 评估插入任务 =====")
        insertion_results = self.evaluate_on_dataset(
            data_dir, task_type="insertion", num_samples=num_dataset_samples
        )
        comprehensive_results["quantitative"]["insertion"] = insertion_results

        # 打印结果
        print("\n插入任务评估结果:")
        print(f"样本数: {insertion_results['samples_evaluated']}")
        print("\n基础模型:")
        print(f"  位置准确率: {insertion_results['base_model']['position_accuracy']:.4f}")
        print(f"  Token准确率: {insertion_results['base_model']['token_accuracy']:.4f}")
        print(f"  组合准确率: {insertion_results['base_model']['combined_accuracy']:.4f}")

        if self.trained_model:
            print("\n训练后模型:")
            print(f"  位置准确率: {insertion_results['trained_model']['position_accuracy']:.4f}")
            print(f"  Token准确率: {insertion_results['trained_model']['token_accuracy']:.4f}")
            print(f"  组合准确率: {insertion_results['trained_model']['combined_accuracy']:.4f}")

        # 量化评估 - 删除任务
        print("\n\n===== 评估删除任务 =====")
        deletion_results = self.evaluate_on_dataset(
            data_dir, task_type="deletion", num_samples=num_dataset_samples
        )
        comprehensive_results["quantitative"]["deletion"] = deletion_results

        # 打印结果
        print("\n删除任务评估结果:")
        print(f"样本数: {deletion_results['samples_evaluated']}")
        print("\n基础模型:")
        print(f"  位置准确率: {deletion_results['base_model']['position_accuracy']:.4f}")
        print(f"  Token准确率: {deletion_results['base_model']['token_accuracy']:.4f}")
        print(f"  组合准确率: {deletion_results['base_model']['combined_accuracy']:.4f}")

        if self.trained_model:
            print("\n训练后模型:")
            print(f"  位置准确率: {deletion_results['trained_model']['position_accuracy']:.4f}")
            print(f"  Token准确率: {deletion_results['trained_model']['token_accuracy']:.4f}")
            print(f"  组合准确率: {deletion_results['trained_model']['combined_accuracy']:.4f}")

        # GSM8K数学数据集评估
        print("\n\n===== 评估数学错误纠正能力 =====")
        try:
            math_results = self.evaluate_on_math_dataset(num_samples=50)
            if math_results:
                comprehensive_results["quantitative"]["math_correction"] = math_results
                print("\n数学错误纠正评估结果:")
                print(f"样本数: {math_results['samples_evaluated']}")
                print(f"基础模型纠错率: {math_results['base_model']['correction_rate']:.4f}")
                if self.trained_model:
                    print(f"训练后模型纠错率: {math_results['trained_model']['correction_rate']:.4f}")
        except Exception as e:
            print(f"评估数学能力时出错: {e}")

        # 重复文本纠正评估
        print("\n\n===== 评估文本重复纠正能力 =====")
        repetition_texts = create_repetition_texts()
        repetition_results = self.evaluate_text_correction(
            repetition_texts, correction_type="repetition", verbose=True
        )
        comprehensive_results["quantitative"]["repetition_correction"] = repetition_results
        print("\n重复文本纠正评估结果:")
        print(f"样本数: {repetition_results['samples_evaluated']}")
        print(f"基础模型纠错率: {repetition_results['base_model']['correction_rate']:.4f}")
        if self.trained_model:
            print(f"训练后模型纠错率: {repetition_results['trained_model']['correction_rate']:.4f}")

        # 质性评估（如果提供了文本样本）
        if qualitative_texts:
            print("\n\n===== 质性评估 =====")
            qualitative_results = self.evaluate_qualitative(qualitative_texts)
            comprehensive_results["qualitative"] = qualitative_results

        # 保存结果
        output_path = os.path.join(output_dir, "evaluation_results.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_results, f, ensure_ascii=False, indent=2)

        print(f"\n评估结果已保存至: {output_path}")

        return comprehensive_results


# 辅助函数
def _contains_repetition(text):
    """检查文本是否包含重复片段"""
    words = text.split()
    for i in range(len(words) - 1):
        if words[i] == words[i + 1]:
            return True
    return False


def create_repetition_texts(num_samples=50):
    """创建包含重复词的文本样本"""
    base_texts = [
        "今天天气真好，我去公园散步。",
        "这个项目需要进行详细的分析。",
        "我喜欢阅读各种类型的书籍。",
        "老师给我们布置了一项作业。",
        "这家餐厅的菜品非常美味。",
        "昨天我参加了一场音乐会。",
        "他正在学习一门新的编程语言。",
        "我们计划下个月去旅行。",
        "这部电影的情节非常感人。",
        "她在研究一个有趣的科学问题。"
    ]

    repetition_texts = []
    for text in base_texts:
        words = text.split()
        for i in range(len(words)):
            # 在随机位置创建重复词
            new_words = words.copy()
            new_words.insert(i, words[i])
            repetition_texts.append(" ".join(new_words))

            # 另一种重复形式
            if i < len(words) - 1:
                new_words = words.copy()
                new_words[i] = words[i] + words[i]
                repetition_texts.append(" ".join(new_words))

    # 随机选择部分样本以达到请求的数量
    if len(repetition_texts) > num_samples:
        import random
        repetition_texts = random.sample(repetition_texts, num_samples)

    return repetition_texts


def introduce_math_errors(text):
    """在文本中引入数学计算错误"""
    import re
    import random

    # 查找文本中的数字
    numbers = re.findall(r'\b\d+\b', text)
    if not numbers:
        return text

    # 随机选择一个数字进行修改
    number_to_change = random.choice(numbers)
    value = int(number_to_change)

    # 根据数值大小进行不同的修改
    if value > 10:
        # 对大数字进行小幅度修改
        new_value = value + random.randint(1, 3)
    else:
        # 对小数字进行+1修改
        new_value = value + 1

    # 替换文本中的数字
    modified_text = text.replace(number_to_change, str(new_value), 1)

    return modified_text


def main():
    parser = argparse.ArgumentParser(description="评估EditLMHF模型的编辑能力")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B", help="基础模型名称")
    parser.add_argument("--trained_model", help="训练后的模型路径")
    parser.add_argument("--data_dir", required=True, help="数据目录")
    parser.add_argument("--output_dir", default="./evaluation_results", help="输出目录")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="设备类型")
    parser.add_argument("--num_samples", type=int, default=500, help="每个评估任务的样本数量")

    # 质性评估相关参数
    parser.add_argument("--qualitative_file", help="包含质性评估文本的文件")
    parser.add_argument("--add_default_texts", action="store_true", help="添加默认的质性评估文本")

    # 特定数据集评估
    parser.add_argument("--evaluate_math", action="store_true", help="评估数学错误纠正能力")
    parser.add_argument("--math_dataset", default="gsm8k", help="数学数据集名称")
    parser.add_argument("--evaluate_repetition", action="store_true", help="评估文本重复纠正能力")
    parser.add_argument("--skip_standard", action="store_true", help="跳过标准的插入/删除评估")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载质性评估文本
    qualitative_texts = []

    if args.add_default_texts:
        default_texts = [
            "这是一个文本，其中有一个明确的错误错误需要修正。",
            "我今天去公园散步了，天气很好很舒适。",
            "她说她将成为医生医生，希望能够帮助更多的人。",
            "编程是一个非常有趣的活动活动，需要逻辑思维。",
            "我们将在明天举办一场会议讨论新项目。"
        ]
        qualitative_texts.extend(default_texts)

    if args.qualitative_file and os.path.exists(args.qualitative_file):
        with open(args.qualitative_file, 'r', encoding='utf-8') as f:
            file_texts = [line.strip() for line in f if line.strip()]
            qualitative_texts.extend(file_texts)

    # 初始化评估器
    evaluator = ModelEvaluator(args.base_model, args.trained_model, args.device)

    # 评估结果存储
    results = {
        "base_model": args.base_model,
        "trained_model": args.trained_model if args.trained_model else "None",
        "quantitative": {},
        "qualitative": None
    }

    # 标准评估
    if not args.skip_standard:
        # 插入任务评估
        print("\n===== 评估插入任务 =====")
        insertion_results = evaluator.evaluate_on_dataset(
            args.data_dir, task_type="insertion", num_samples=args.num_samples
        )
        results["quantitative"]["insertion"] = insertion_results

        # 打印结果
        print("\n插入任务评估结果:")
        print(f"样本数: {insertion_results['samples_evaluated']}")
        print(f"基础模型组合准确率: {insertion_results['base_model']['combined_accuracy']:.4f}")
        if evaluator.trained_model:
            print(f"训练后模型组合准确率: {insertion_results['trained_model']['combined_accuracy']:.4f}")

        # 删除任务评估
        print("\n===== 评估删除任务 =====")
        deletion_results = evaluator.evaluate_on_dataset(
            args.data_dir, task_type="deletion", num_samples=args.num_samples
        )
        results["quantitative"]["deletion"] = deletion_results

        # 打印结果
        print("\n删除任务评估结果:")
        print(f"样本数: {deletion_results['samples_evaluated']}")
        print(f"基础模型组合准确率: {deletion_results['base_model']['combined_accuracy']:.4f}")
        if evaluator.trained_model:
            print(f"训练后模型组合准确率: {deletion_results['trained_model']['combined_accuracy']:.4f}")

    # 数学能力评估
    if args.evaluate_math:
        print("\n===== 评估数学错误纠正能力 =====")
        try:
            math_results = evaluator.evaluate_on_math_dataset(
                dataset_name=args.math_dataset, num_samples=args.num_samples // 10
            )
            if math_results:
                results["quantitative"]["math_correction"] = math_results
                print("\n数学错误纠正评估结果:")
                print(f"样本数: {math_results['samples_evaluated']}")
                print(f"基础模型纠错率: {math_results['base_model']['correction_rate']:.4f}")
                if evaluator.trained_model:
                    print(f"训练后模型纠错率: {math_results['trained_model']['correction_rate']:.4f}")
        except Exception as e:
            print(f"评估数学能力时出错: {e}")

    # 重复文本纠正评估
    if args.evaluate_repetition:
        print("\n===== 评估文本重复纠正能力 =====")
        repetition_texts = create_repetition_texts(num_samples=min(50, args.num_samples))
        repetition_results = evaluator.evaluate_text_correction(
            repetition_texts, correction_type="repetition", verbose=True
        )
        results["quantitative"]["repetition_correction"] = repetition_results
        print("\n重复文本纠正评估结果:")
        print(f"样本数: {repetition_results['samples_evaluated']}")
        print(f"基础模型纠错率: {repetition_results['base_model']['correction_rate']:.4f}")
        if evaluator.trained_model:
            print(f"训练后模型纠错率: {repetition_results['trained_model']['correction_rate']:.4f}")

    # 质性评估
    if qualitative_texts:
        print("\n===== 质性评估 =====")
        qualitative_results = evaluator.evaluate_qualitative(qualitative_texts)
        results["qualitative"] = qualitative_results

    # 保存结果
    output_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n评估结果已保存至: {output_path}")

    # 如果有训练后的模型，打印性能提升摘要
    if evaluator.trained_model and not args.skip_standard:
        print("\n\n===== 性能提升摘要 =====")

        # 插入任务
        insertion_base = results["quantitative"]["insertion"]["base_model"]["combined_accuracy"]
        insertion_trained = results["quantitative"]["insertion"]["trained_model"]["combined_accuracy"]
        insertion_improvement = insertion_trained - insertion_base

        # 删除任务
        deletion_base = results["quantitative"]["deletion"]["base_model"]["combined_accuracy"]
        deletion_trained = results["quantitative"]["deletion"]["trained_model"]["combined_accuracy"]
        deletion_improvement = deletion_trained - deletion_base

        print(f"插入任务准确率提升: {insertion_improvement:.4f} ({insertion_improvement * 100:.2f}%)")
        print(f"删除任务准确率提升: {deletion_improvement:.4f} ({deletion_improvement * 100:.2f}%)")
        print(
            f"平均准确率提升: {(insertion_improvement + deletion_improvement) / 2:.4f} ({(insertion_improvement + deletion_improvement) * 50:.2f}%)")

        # 其他任务提升（如果有评估）
        if args.evaluate_repetition and "repetition_correction" in results["quantitative"]:
            rep_base = results["quantitative"]["repetition_correction"]["base_model"]["correction_rate"]
            rep_trained = results["quantitative"]["repetition_correction"]["trained_model"]["correction_rate"]
            rep_improvement = rep_trained - rep_base
            print(f"重复文本纠正能力提升: {rep_improvement:.4f} ({rep_improvement * 100:.2f}%)")

        if args.evaluate_math and "math_correction" in results["quantitative"]:
            math_base = results["quantitative"]["math_correction"]["base_model"]["correction_rate"]
            math_trained = results["quantitative"]["math_correction"]["trained_model"]["correction_rate"]
            math_improvement = math_trained - math_base
            print(f"数学错误纠正能力提升: {math_improvement:.4f} ({math_improvement * 100:.2f}%)")


if __name__ == "__main__":
    main()