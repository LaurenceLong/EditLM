# EditLM: 高效文本编辑语言模型

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

EditLM 是一个旨在高效执行文本编辑任务的语言模型。与需要完全微调整个大型语言模型（LLM）的传统方法不同，EditLM 采用了一种更轻量级的方法，通过在预训练的 LLM 基础上增加专门的编辑组件，并利用特定的编辑任务进行训练，从而实现对文本的精确修改（如插入、删除）。

## 项目方案概述

**核心思想:**

1.  **利用预训练模型:** EditLM 建立在一个强大的预训练语言模型（如 Qwen2.5-0.5B）之上，利用其已有的语言理解和生成能力。
2.  **专门的编辑组件:** 模型引入了新的可学习组件，包括：
    *   `GAP` 令牌嵌入：代表潜在的编辑点（插入或删除的位置）。
    *   `Boundary` 嵌入：用于处理序列边界情况。
    *   投影层 (`triple_proj`, `fuse_proj`)：用于融合局部上下文和全局信息。
    *   编辑头 (`index_head`, `edit_head`)：`index_head` 预测编辑发生的位置（0 到 L，其中 L 表示序列末尾），`edit_head` 预测要插入的令牌或指示删除操作。
3.  **局部与全局信息融合:**
    *   **局部上下文:** 对于每个潜在的编辑点（GAP），模型考虑其左右相邻令牌的表示，形成 `[左侧上下文, GAP嵌入, 右侧上下文]` 的三元组表示。
    *   **全局上下文:** 为了让编辑决策考虑到整个序列的信息，EditLM 巧妙地**共享并利用**了基础模型最后一层自注意力机制的 Q, K, V, O 投影权重。它将 GAP 的局部表示作为查询（Query），将整个序列的隐藏状态作为键（Key）和值（Value），计算出一个全局上下文表示。模型特别处理了分组查询注意力（GQA）和多查询注意力（MQA）的情况。
    *   **融合:** 局部三元组表示和计算得到的全局上下文表示被融合在一起，形成最终用于预测编辑位置和内容的 GAP 状态。
4.  **专门的训练任务:** 模型通过三种特定任务进行训练，以学习不同的编辑能力：
    *   **预测 (Prediction):** 标准的下一个词预测任务，维持模型的语言生成能力。目标索引指向序列末尾 (`L`)，目标令牌是序列的下一个词。
    *   **删除 (Deletion):** 在原始序列的随机位置**插入**一个随机令牌，然后训练模型在该位置预测一个特殊的**删除令牌**（本项目中使用 `pad_token_id`）。这教会模型识别并标记需要删除的内容。目标索引指向插入的随机令牌位置，目标令牌是删除令牌 ID。
    *   **插入 (Insertion):** 从原始序列中随机**删除**一个令牌，然后训练模型在被删除的位置预测**原始被删除的令牌**。这教会模型根据上下文恢复丢失的内容。目标索引指向删除发生的位置，目标令牌是被删除的原始令牌。
5.  **高效训练:** 可以选择冻结基础模型的参数（`--freeze`），只训练新增的编辑组件和少量相关层，大大减少了计算资源需求和训练时间。

**优势:**

*   **高效:** 相比完全微调，训练成本更低。
*   **精确编辑:** 专门为编辑任务设计，能够更精确地定位和执行修改。
*   **保留生成能力:** 通过预测任务和共享基础模型权重，保留了原始模型的语言理解和生成能力。

## 项目结构

```
EditLM/
├── README.md                 # 本文档
├── requirements.txt          # 项目依赖
└── src/                      # 源代码目录
    ├── __init__.py
    ├── config.py             # 模型和训练配置（目前主要在脚本参数中设置）
    ├── model_hf.py           # EditLM 模型定义 (基于 Hugging Face Transformers)
    ├── preprocess_data.py    # 数据预处理脚本 (WikiText-103 -> 编辑任务)
    ├── tokenizer.py          # Tokenizer 获取工具 (包装 HF Tokenizer)
    ├── train_sft_hf.py       # 主要的训练脚本
    ├── inference.py          # 推理脚本
    └── utils.py              # 辅助函数 (检查点保存/加载, 学习率调度器)
```

## 安装

1.  **克隆仓库:**
    ```bash
    git clone <your-repo-url> # 替换为你的仓库URL
    cd EditLM
    ```

2.  **创建虚拟环境 (推荐):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    # 或者 .venv\Scripts\activate # Windows
    ```

3.  **安装依赖:**
    ```bash
    pip install -r requirements.txt
    ```

## 数据准备

训练 EditLM 需要经过特殊预处理的数据集。`src/preprocess_data.py` 脚本用于将标准文本数据集（如此处使用的 WikiText-103）转换为模型所需的格式，包含预测、删除和插入三种任务类型。

**预处理步骤:**

1.  **运行预处理脚本:**
    ```bash
    python src/preprocess_data.py \
        --base "Qwen/Qwen2.5-0.5B" \
        --output_dir "./data/wikitext_processed" \
        --seq_len 256 \
        --chunk_size 1000 \
        --prediction_ratio 0.05 \
        --deletion_ratio 0.05 \
        --insertion_ratio 0.05 \
        --sample_rate 0.1 # 可选：为了快速测试，只处理10%的原始数据
    ```

    *   `--base`: 指定基础模型，用于加载对应的 Tokenizer。
    *   `--output_dir`: 指定处理后数据的存放目录。
    *   `--seq_len`: 定义模型处理的序列长度。
    *   `--chunk_size`: 每个二进制数据块包含的样本数量。
    *   `--prediction_ratio`, `--deletion_ratio`, `--insertion_ratio`: 控制为每个原始序列生成相应任务样本的比例（近似）。
    *   `--sample_rate`: 从原始数据集中采样的比例，用于减少生成的数据量，加快预处理速度（尤其适用于大型数据集的初步实验）。

2.  **理解输出:**
    脚本会在 `--output_dir` 下生成三个子目录：`prediction`, `deletion`, `insertion`，分别包含对应任务的数据块（`.pt` 文件）和元数据（`_metadata.pt`）。
    *   `prediction/`: 包含预测任务数据。输入是序列前缀，目标索引是序列长度，目标令牌是下一个词。
    *   `deletion/`: 包含删除任务数据。输入是插入了随机词的序列，目标索引是插入位置，目标令牌是特殊的删除符 (pad_token_id)。
    *   `insertion/`: 包含插入任务数据。输入是删除了某个词的序列，目标索引是删除位置，目标令牌是被删除的原始词。

## 训练

使用 `src/train_sft_hf.py` 脚本进行模型训练。训练过程会交替从 `prediction`（如果未跳过）、`deletion` 和 `insertion` 目录加载数据批次。

**训练命令示例:**

### 1. 完整训练 (包含预测任务和编辑任务)

这是最全面的训练方式，同时训练模型的生成能力和编辑能力。

```bash
python src/train_sft_hf.py \
    --base "Qwen/Qwen2.5-0.5B" \
    --outdir "./checkpoints/editlm_qwen_full_train" \
    --data_dir "./data/wikitext_processed" \
    --batch_size 8 \
    --steps 100000 \
    --ckpt_every 5000 \
    --lr 2e-4 \
    --grad_accum 4 # 梯度累积步数
```

### 2. 冻结 Backbone 训练 (节省资源)

冻结基础模型的参数，只训练新增的编辑相关组件。这可以显著降低 GPU 内存需求并加快训练速度，适用于计算资源有限的情况。

```bash
python src/train_sft_hf.py \
    --base "Qwen/Qwen2.5-0.5B" \
    --outdir "./checkpoints/editlm_qwen_frozen" \
    --data_dir "./data/wikitext_processed" \
    --batch_size 8 \
    --steps 50000 \
    --freeze # 冻结基础模型参数
```

### 3. 仅编辑任务训练 (跳过预测任务)

如果基础模型已经足够强大，或者你想专注于编辑能力的训练，可以跳过预测任务。

```bash
# 从头开始只训练编辑任务
python src/train_sft_hf.py \
    --base "Qwen/Qwen2.5-0.5B" \
    --outdir "./checkpoints/editlm_qwen_edit_only_scratch" \
    --data_dir "./data/wikitext_processed" \
    --batch_size 8 \
    --steps 50000 \
    --skip_prediction # 跳过预测任务

# 或者，从一个已有的检查点（可能是只训练了预测任务的模型）开始，只进行编辑任务训练
# 假设 './checkpoints/prediction_model/pred_final.pt' 是一个预训练好的模型检查点
python src/train_sft_hf.py \
    --from_file ./checkpoints/prediction_model/pred_final.pt \
    --outdir "./checkpoints/editlm_qwen_edit_only_from_ckpt" \
    --data_dir "./data/wikitext_processed" \
    --batch_size 8 \
    --steps 50000 \
    --skip_prediction # 跳过预测任务
```

### 4. 恢复训练

如果训练意外中断，可以使用 `--from_file` 参数从最后一个保存的检查点恢复训练状态（包括模型权重、优化器状态和训练步数）。

```bash
python src/train_sft_hf.py \
    --from_file "./checkpoints/editlm_qwen_frozen/sft_step25000.pt" \
    --outdir "./checkpoints/editlm_qwen_frozen" \
    --data_dir "./data/wikitext_processed" \
    --batch_size 8 \
    --steps 50000 \
    --freeze # 确保恢复时使用与之前相同的配置 (如 --freeze)
```

**关键训练参数说明:**

*   `--base`: (必需) 使用的基础模型名称或路径 (Hugging Face Hub 或本地)。
*   `--outdir`: (必需) 训练检查点和最终模型的保存目录。
*   `--data_dir`: (必需) 包含预处理后 `prediction/`, `deletion/`, `insertion/` 子目录的数据根目录。
*   `--from_file`: (可选) 从指定的检查点文件 (`.pt`) 加载模型、优化器状态和步数以恢复训练。
*   `--batch_size`: 每个 GPU 上每个任务的批次大小。有效批次大小为 `batch_size * grad_accum`。根据 GPU 内存调整。
*   `--steps`: 训练总步数 (所有任务的总和)。
*   `--freeze`: (可选标志) 如果设置，则冻结基础模型的参数，只训练编辑相关的头部和嵌入层。
*   `--skip_prediction`: (可选标志) 如果设置，则训练循环将不包含预测任务，只进行删除和插入任务的训练。
*   `--lr`: 学习率峰值。
*   `--grad_accum`: 梯度累积步数。将梯度累积 `N` 步后执行一次优化器更新。用于在不增加内存的情况下模拟更大的批次大小。
*   `--warmup_steps`: 学习率预热步数。
*   `--ckpt_every`: 每隔多少步保存一次检查点。
*   `--seed`: 随机种子，用于保证实验可复现性。

**内存不足时的建议:**

1.  **使用 `--freeze`:** 这是最有效的方法，显著减少需要存储梯度和优化器状态的参数量。
2.  **减小 `--batch_size`:** 降低每个批次处理的样本数（例如改为 4 或 2）。
3.  **增加 `--grad_accum`:** 配合减小的 `batch_size`，通过累积更多步数的梯度来维持等效的有效批次大小，但这会减慢训练速度（因为优化器更新频率降低）。
4.  **使用更小的基础模型:** 如果上述方法仍不足，考虑更换 `--base` 为更小的模型（如 `Qwen/Qwen1.5-0.5B` 或其他更小的模型）。

**示例 (低内存配置):**

```bash
python src/train_sft_hf.py \
    --base "Qwen/Qwen2.5-0.5B" \
    --outdir "./checkpoints/editlm_qwen_lowmem" \
    --data_dir "./data/wikitext_processed" \
    --batch_size 2 \
    --grad_accum 8 \
    --steps 50000 \
    --freeze
```

### 定时任务 (Linux `at` 命令示例)

如果你想在特定时间自动开始训练（例如，从一个预训练好的预测模型开始编辑任务训练）：

```bash
# 注意: 路径需要根据你的实际环境修改
# 将命令和重定向包装在引号内传递给 echo
echo "cd /path/to/your/EditLM && \
/path/to/your/.venv/bin/python src/train_sft_hf.py \
--from_file ./checkpoints/editlm_qwen_frozen/prediction_phase.pt \
--outdir \"./checkpoints/editlm_qwen_edit_only\" \
--data_dir \"./src/data/wikitext_processed\" \
--batch_size 8 \
--steps 50000 \
--skip_prediction \
> training_scheduled.log 2>&1" | at 09:00
```

## 推理

使用 `src/inference.py` 脚本加载训练好的 EditLM 模型进行文本编辑或生成。

**推理模式:**

*   **单次编辑 (`--single_edit`):** 模型接收输入文本，预测一个编辑操作（位置和内容/删除符），执行该操作，然后停止。适用于纠错等单步修改场景。
*   **连续编辑/生成 (默认):** 模型接收输入文本，预测并执行第一个编辑操作。然后将修改后的文本再次输入模型，进行下一个编辑操作，如此循环，直到达到最大编辑次数 (`--max_edits`) 或最大序列长度 (`--max_len`)。适用于文本续写、润色等需要多次修改或生成内容的场景。

**推理命令示例:**

### 1. 对单个输入文本进行单次编辑

```bash
python src/inference.py \
    --model_path "./checkpoints/editlm_qwen_edit_only/sft_final_50000.pt" \
    --input_text "这是一个文本，其中有一个明确的错误错误需要修正。" \
    --single_edit
```

### 2. 对单个输入文本进行连续编辑 (自动生成/补全)

```bash
python src/inference.py \
    --model_path "./checkpoints/editlm_qwen_edit_only/sft_final_50000.pt" \
    --input_text "故事开始于一个黑暗的夜晚，" \
    --max_len 100 \
    --max_edits 30 # 限制编辑/生成步数
```

### 3. 从文件批量推理 (每行一个文本)

```bash
# sample_texts.txt 文件内容示例:
# 这是第一行文本。
# 这是第二行，可能需要编辑。
# 第三行。

python src/inference.py \
    --model_path "./checkpoints/editlm_qwen_edit_only/sft_final_50000.pt" \
    --input_file "sample_texts.txt" \
    --output_file "batch_results.json" \
    --max_len 150 \
    --max_edits 20
```

### 4. 从 Hugging Face 数据集推理

```bash
python src/inference.py \
    --model_path "./checkpoints/editlm_qwen_edit_only/sft_final_50000.pt" \
    --dataset "gsm8k" \
    --dataset_split "test" \
    --text_column "question" \
    --max_samples 10 \
    --output_dir "./results_gsm8k" \
    --max_len 200
```

### 5. 从自定义 JSON/CSV/Parquet 文件推理

```bash
# custom_data.json 文件内容示例 (JSON Lines):
# {"id": 1, "text": "需要编辑的文本一。"}
# {"id": 2, "text": "需要编辑的文本二。"}

python src/inference.py \
    --model_path "./checkpoints/editlm_qwen_edit_only/sft_final_50000.pt" \
    --dataset_path "custom_data.json" \
    --text_column "text" \
    --output_dir "./custom_results" \
    --max_len 128
```

**关键推理参数说明:**

*   `--model_path`: (必需) 指向训练好的 EditLM 检查点文件 (`.pt`) 的路径。脚本会自动尝试从同目录加载 Tokenizer，如果找不到会回退使用模型检查点中记录的基础模型名称加载。
*   `--input_text`: (可选) 直接提供单个输入文本。
*   `--input_file`: (可选) 提供一个文本文件路径，每行包含一个待处理的文本。
*   `--dataset`: (可选) Hugging Face 数据集名称 (例如 `gsm8k`, `wikitext`)。
*   `--dataset_split`: (可选, 需配合 `--dataset`) 使用的数据集划分 (例如 `test`, `validation`)。
*   `--dataset_path`: (可选) 指向本地数据集文件的路径 (支持 `.json`, `.csv`, `.parquet`)。
*   `--text_column`: (必需, 当使用 `--dataset` 或 `--dataset_path` 时) 数据集中包含文本内容的列名。
*   `--output_file`: (可选, 用于 `--input_file`) 将批量推理结果保存为 JSON Lines 文件。
*   `--output_dir`: (可选, 用于数据集推理) 将结果保存到指定目录（每个样本一个 JSON 文件）。
*   `--max_samples`: (可选, 用于数据集推理) 处理的最大样本数量。
*   `--max_len`: (可选) 生成或编辑后的最大序列长度。
*   `--max_edits`: (可选, 仅用于连续编辑模式) 最大执行的编辑/生成步数。
*   `--single_edit`: (可选标志) 如果设置，则执行单次编辑模式，否则执行连续编辑/生成模式。
*   `--temperature`: (可选, 默认 1.0) 控制采样随机性，较低的值使输出更确定。
*   `--top_k`: (可选, 默认 50) 限制采样时只考虑概率最高的 K 个词。

## (可选) 评估

当前版本的 `inference.py` 可以用于定性评估模型的编辑效果。对于定量的自动化评估（例如，与参考编辑进行比较，计算 BLEU、ROUGE 或编辑距离），需要额外的脚本或集成评估框架。你可以修改 `inference.py` 或编写新的脚本，将模型的输出与期望的编辑结果进行比较。

## 贡献

欢迎提交问题 (Issues) 和拉取请求 (Pull Requests)。

## 许可证

本项目采用 [MIT 许可证](LICENSE)。