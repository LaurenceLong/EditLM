# README.md


## 数据集
### 测试数据集
``` bash
python src/test_tokenization.py --data_dir "src/data/wikitext_processed" --base "Qwen/Qwen2.5-0.5B" --num_samples 3
```

## 训练
以下是启动 `train_sft_hf.py` 训练脚本的命令，根据不同的训练需求有几种常见组合：

### 1. 基本训练命令（包含预测任务和编辑任务）

```bash
python src/train_sft_hf.py --base "Qwen/Qwen2.5-0.5B" --outdir "./checkpoints/editlm_qwen" --data_dir "./src/data/wikitext_processed" --batch_size 8 --steps 50000
```

### 2. 只训练新增的头部，冻结backbone参数（节省GPU内存，加快训练）

```bash
python src/train_sft_hf.py --base "Qwen/Qwen2.5-0.5B" --outdir "./checkpoints/editlm_qwen_frozen" --data_dir "./src/data/wikitext_processed" --batch_size 8 --steps 50000 --freeze
```

### 3. 跳过预测任务，只进行编辑任务训练（适用于已有预训练模型）

```bash
python src/train_sft_hf.py --base "Qwen/Qwen2.5-0.5B" --outdir "./checkpoints/editlm_qwen_edit_only" --data_dir "./src/data/wikitext_processed" --batch_size 8 --steps 50000 --skip_prediction
# 从预先报寸的模型加载训练
python src/train_sft_hf.py --from_file ./checkpoints/editlm_qwen_frozen/prediction_phase.pt --outdir "./checkpoints/editlm_qwen_edit_only" --data_dir "./src/data/wikitext_processed" --batch_size 8 --steps 50000 --skip_prediction
```

### 4. 完整控制训练过程的命令

```bash
python src/train_sft_hf.py --base "Qwen/Qwen2.5-0.5B" --outdir "./checkpoints/editlm_qwen_full" --data_dir "./src/data/wikitext_processed" --batch_size 16 --steps 100000 --freeze --prediction_epochs 2
```

### 参数说明

- `--base`: 使用的基础模型名称，可以是HuggingFace上的模型名或本地路径
- `--outdir`: 模型保存目录
- `--data_dir`: 预处理数据目录
- `--batch_size`: 训练批次大小，根据GPU内存调整
- `--steps`: 训练总步数
- `--freeze`: 如果添加此参数，将冻结backbone，只训练新增的头部
- `--skip_prediction`: 如果添加此参数，将跳过预测任务训练
- `--prediction_epochs`: 预测任务训练的轮数，默认为1

### 内存不足时的建议

如果遇到GPU内存不足的问题，可以：

1. 使用 `--freeze` 参数冻结backbone
2. 减小 `--batch_size` 的值（例如改为4或2）
3. 结合梯度累积（代码中已设置为4步累积一次）

```bash
python src/train_sft_hf.py --base "Qwen/Qwen2.5-0.5B" --outdir "./checkpoints/editlm_qwen_lowmem" --data_dir "./src/data/wikitext_processed" --batch_size 2 --steps 50000 --freeze
```

### 使用更小的模型

如果GPU内存仍然不足，可以考虑使用更小的基础模型：

```bash
python src/train_sft_hf.py --base "Qwen/Qwen1.5-0.5B" --outdir "./checkpoints/editlm_small" --data_dir "./src/data/wikitext_processed" --batch_size 4 --steps 50000 --freeze
```

确保在执行命令前，已经完成了数据预处理步骤，并且路径指向正确的数据目录。

### 定时任务
```cmd
echo "cd /home/laurence/work/ai/EditLM && /home/laurence/work/ai/EditLM/.venv/bin/python src/train_sft_hf.py --from_file ./checkpoints/editlm_qwen_frozen/prediction_phase.pt --outdir \"./checkpoints/editlm_qwen_edit_only\" --data_dir \"./src/data/wikitext_processed\" --batch_size 8 --steps 50000 --skip_prediction > training_scheduled.log 2>&1" | at 09:00
```


## 推理

提供了多种输入方式来进行文本编辑推理：

```bash
# 单个文本推理
python src/inference.py --model_path "./checkpoints/editlm_qwen_edit_only/sft_final_50000.pt" --input_text "这是一个文本，其中有一个明确的错误错误需要修正。"

# 从文件批量推理
python src/inference.py --model_path "./checkpoints/editlm_qwen_edit_only/sft_final_50000.pt" --input_file "sample_texts.txt"

# 从HuggingFace数据集推理
python src/inference.py --model_path "./checkpoints/editlm_qwen_edit_only/sft_final_50000.pt" --dataset "gsm8k" --dataset_split "test" --text_column "question" --max_samples 100

# 从自定义数据集文件推理
python src/inference.py --model_path "./checkpoints/editlm_qwen_edit_only/sft_final_50000.pt" --dataset_path "custom_data.json" --text_column "text"
```

## 评估

以下是使用修改后代码进行推理的几种常用命令示例：

### 1. 单个文本的单次编辑推理

```bash
python src/inference.py --model_path "./checkpoints/editlm_qwen_edit_only/sft_final_50000.pt" --input_text "这是一个示例文本，等待模型进行编辑。" --single_edit
```

### 2. 单个文本的连续编辑（自动生成完整文本）

```bash
python src/inference.py --model_path "./checkpoints/editlm_qwen_edit_only/sft_final_50000.pt" --input_text "这是一个短文本开头，" --max_len 100
```

### 3. 限制最大生成长度的推理

```bash
python src/inference.py --model_path "./checkpoints/editlm_qwen_edit_only/sft_final_50000.pt" --input_text "请继续生成这个故事：从前，有一个" --max_len 50 --max_edits 15
```

### 4. 从文件批量处理多个文本

```bash
python src/inference.py --model_path "./checkpoints/editlm_qwen_edit_only/sft_final_50000.pt" --input_file "sample_texts.txt" --max_len 200 --output_file "batch_results.json"
```

### 5. 从HuggingFace数据集加载文本并推理

```bash
python src/inference.py --model_path "./checkpoints/editlm_qwen_edit_only/sft_final_50000.pt" --dataset "gsm8k" --dataset_split "test" --text_column "question" --max_samples 5 --max_len 300
```

### 6. 使用自定义数据集并指定输出目录

```bash
python src/inference.py --model_path "./checkpoints/editlm_qwen_edit_only/sft_final_50000.pt" --dataset_path "custom_data.json" --text_column "text" --output_dir "./custom_results" --max_len 150
```