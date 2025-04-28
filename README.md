# EditLM: 高效文本编辑语言模型

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

EditLM 是一个旨在高效执行文本编辑任务的语言模型。与需要完全微调整个大型语言模型（LLM）的传统方法不同，EditLM 采用了一种轻量级方法，在预训练模型（例如 Qwen 系列）的基础上增加专门的编辑组件，针对插入、删除和预测等任务进行微调，从而实现对文本的精准编辑和纠错。

---

## 项目方案概述

### 核心思想

1. **利用预训练模型**  
   EditLM 建立在一个强大的预训练语言模型（例如 Qwen/Qwen2.5-0.5B）之上，充分利用其自然语言理解与生成能力。

2. **专门的编辑组件**  
   为应对编辑任务，EditLM 引入了新的可学习模块，包括：
   - **GAP 令牌嵌入**：表示潜在的编辑点。
   - **Boundary 嵌入**：处理序列边界问题。
   - **局部投影层**（`triple_proj`、`fuse_proj`）：融合左右上下文与全局信息。
   - **编辑头**（`index_head`、`edit_head`）：分别负责预测编辑位置以及预测插入或删除的 token。
   
3. **局部与全局信息融合**  
   - **局部上下文**：利用左侧令牌、GAP 嵌入和右侧令牌构造局部表示。
   - **全局上下文**：共享预训练模型最后层自注意力机制的 Q/K/V/O 参数用于计算全局上下文。
   - **融合**：将局部表示和全局上下文经过融合层后用于编辑任务的预测。

4. **专门的训练任务**  
   模型通过三种任务进行训练：
   - **预测任务**：标准下一个词预测，确保语言生成能力。
   - **删除任务**：在原始序列中插入随机 token，然后训练模型预测一个特殊删除令牌（使用 pad_token_id）。
   - **插入任务**：从原始序列中删除一个 token，训练模型恢复删除的内容。

5. **高效训练**  
   通过冻结基础模型参数（使用 `--freeze` 参数）仅训练新增编辑组件、使用 FP16 以及梯度累积等手段，大幅降低内存和计算资源的需求。

### 优势

- **高效**：相比从头全量微调，训练成本更低。
- **精确编辑**：专门设计的编辑模块能够定位并精确修改文本。
- **保持生成能力**：在编辑训练同时，保留原有预训练模型的语言生成能力。

---

## 项目结构

```
EditLM/
├── README.md                  # 本文档
├── requirements.txt           # 项目依赖
└── src/                       # 源代码目录
    ├── __init__.py
    ├── config.py              # 模型和训练配置
    ├── data.py                # 数据集加载与预处理（删除、插入、预测任务）
    ├── evaluate.py            # 模型评估脚本（定量与质性评估）
    ├── inference.py           # 推理脚本：支持单次及连续编辑模式
    ├── model_hf.py            # EditLM 模型定义（基于 Hugging Face Transformers）
    ├── preprocess_data.py     # 数据预处理脚本：生成编辑任务数据
    ├── test_hf_model.py       # 测试脚本：通过命令行测试文本生成管道
    ├── test_tokenization.py   # 测试脚本：验证预处理数据和 tokenization
    ├── tokenizer.py           # Tokenizer 工具封装（获取删除token ID等）
    ├── train_sft_hf.py        # 训练脚本：支持冻结backbone、梯度累积及检查点管理
    └── utils.py               # 辅助函数：检查点保存/加载、学习率调度器等
```

---

## 安装

1. **克隆仓库：**

   ```bash
   git clone <your-repo-url>  # 替换为你的仓库URL
   cd EditLM
   ```

2. **创建虚拟环境（推荐）：**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   # 或者 .venv\Scripts\activate  # Windows
   ```

3. **安装依赖：**

   ```bash
   pip install -r requirements.txt
   ```

---

## 数据准备

预处理脚本 `src/preprocess_data.py` 可将 WikiText-103 数据集转换为模型所需格式，生成包含预测、删除和插入任务的二进制数据文件。

**示例命令：**

```bash
python src/preprocess_data.py \
    --base "Qwen/Qwen2.5-0.5B" \
    --seq_len 256 \
    --output_dir "./data/wikitext_processed" \
    --chunk_size 1000 \
    --prediction_ratio 0.05 \
    --deletion_ratio 0.05 \
    --insertion_ratio 0.05 \
    --sample_rate 0.1    # 可选：减少处理样本量以便快速测试
```

预处理完成后，`./data/wikitext_processed` 下会生成 `prediction/`、`deletion/` 和 `insertion/` 三个子目录以及对应的元数据文件。

---

## 训练

使用 `src/train_sft_hf.py` 脚本训练模型，支持全量训练、冻结基础模型以及跳过预测任务等配置。

**示例：**

1. **完整训练（包含预测和编辑任务）：**

   ```bash
   python src/train_sft_hf.py \
       --base "Qwen/Qwen2.5-0.5B" \
       --outdir "./checkpoints/editlm_qwen_full_train" \
       --data_dir "./data/wikitext_processed" \
       --batch_size 8 \
       --steps 100000 \
       --ckpt_every 5000 \
       --lr 2e-4 \
       --grad_accum 4
   ```

2. **冻结 Backbone 训练：**

   ```bash
   python src/train_sft_hf.py \
       --base "Qwen/Qwen2.5-0.5B" \
       --outdir "./checkpoints/editlm_qwen_frozen" \
       --data_dir "./data/wikitext_processed" \
       --batch_size 8 \
       --steps 50000 \
       --freeze
   ```

3. **仅编辑任务训练（跳过预测任务）：**

   ```bash
   python src/train_sft_hf.py \
       --base "Qwen/Qwen2.5-0.5B" \
       --outdir "./checkpoints/editlm_qwen_edit_only" \
       --data_dir "./data/wikitext_processed" \
       --batch_size 8 \
       --steps 50000 \
       --skip_prediction
   ```

4. **恢复训练：**

   ```bash
   python src/train_sft_hf.py \
       --from_file "./checkpoints/editlm_qwen_frozen/sft_step25000.pt" \
       --outdir "./checkpoints/editlm_qwen_frozen" \
       --data_dir "./data/wikitext_processed" \
       --batch_size 8 \
       --steps 50000 \
       --freeze
   ```

**内存不足建议：**

- 使用 `--freeze` 冻结预训练模型参数；
- 降低 `--batch_size` 或增加 `--grad_accum`；
- 如有必要，可选择更小的基础模型。

---

## 推理

使用 `src/inference.py` 脚本进行文本编辑推理，可选择单次编辑或连续编辑（迭代编辑直到达到最大编辑次数或序列长度）。

**示例：**

1. **单次编辑：**

   ```bash
   python src/inference.py \
       --model_path "./checkpoints/editlm_qwen_edit_only/sft_final_50000.pt" \
       --input_text "这是需要修正的文本，其中包含错误错误。" \
       --single_edit
   ```

2. **连续编辑（自动生成）：**

   ```bash
   python src/inference.py \
       --model_path "./checkpoints/editlm_qwen_edit_only/sft_final_50000.pt" \
       --input_text "故事开始于一个黑暗的夜晚，" \
       --max_len 100 \
       --max_edits 30
   ```

3. **批量推理（从文件）：**

   ```bash
   python src/inference.py \
       --model_path "./checkpoints/editlm_qwen_edit_only/sft_final_50000.pt" \
       --input_file "sample_texts.txt" \
       --output_file "batch_results.json" \
       --max_len 150 \
       --max_edits 20
   ```

4. **从数据集推理：**

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

---

## 评估

使用 `src/evaluate.py` 脚本对模型进行综合评估，支持：

- **定量评估**：在插入、删除任务上计算位置、token 与组合准确率；对数学错误与重复文本的纠正能力进行评估；
- **质性评估**：对指定文本进行编辑并生成详细的编辑历史报告。

**示例命令：**

```bash
python src/evaluate.py \
    --base_model "Qwen/Qwen2.5-0.5B" \
    --trained_model "./checkpoints/editlm_qwen_edit_only/sft_final_50000.pt" \
    --data_dir "./data/wikitext_processed" \
    --output_dir "./evaluation_results" \
    --num_samples 500 \
    --evaluate_math \
    --evaluate_repetition \
    --add_default_texts
```

评估结果将保存为 `evaluation_results/evaluation_results.json` ，同时生成摘要报告文件。

---

## 测试

项目提供两个测试脚本用于验证关键模块功能：

1. **测试文本生成（模型推理）：**  
   使用 `src/test_hf_model.py` 通过命令行指定输入与模型名称进行文本生成测试。

   **示例：**

   ```bash
   python src/test_hf_model.py --content "请检查以下计算：12 + 13 = 42" --model "Qwen/Qwen2.5-1.5B-Instruct"
   ```

2. **测试 Tokenization 与预处理数据：**  
   使用 `src/test_tokenization.py` 测试数据集加载、tokenizer 效果以及编辑任务数据转换正确性。

   **示例：**

   ```bash
   python src/test_tokenization.py --data_dir "./data/wikitext_processed" --base "Qwen/Qwen2.5-0.5B" --num_samples 5
   ```

---

## 贡献

欢迎提交问题（Issues）和合并请求（Pull Requests）以改进项目。如果你有新的想法或改进方案，请随时参与贡献。

---

## 许可证

本项目采用 [MIT 许可证](LICENSE) 授权。

---及测试使用方法，方便用户按照项目说明快速上手。