"""配置文件"""

class Config:
    # 基础配置
    model_name_or_path = "facebook/opt-125m"  # 使用OPT-125M小模型替代Llama-2-7b
    output_dir = "./output"
    seed = 42

    # 数据配置
    train_file = "path/to/train.json"  # 训练数据路径
    eval_file = "path/to/eval.json"    # 评估数据路径
    max_seq_length = 512               # 最大序列长度
    preprocessing_num_workers = 4

    # 训练配置
    learning_rate = 5e-5
    weight_decay = 0.01
    batch_size = 16      # 更小的模型可以使用更大的批次大小
    gradient_accumulation_steps = 2
    num_train_epochs = 3
    warmup_ratio = 0.1
    logging_steps = 100
    eval_steps = 500
    save_steps = 1000

    # 模型配置
    index_loss_weight = 1.0  # 索引损失的权重系数α
    lm_loss_weight = 0.5     # 语言模型损失的权重系数β (用于预训练)

    # 推理配置
    max_edit_steps = 50  # 推理时最大编辑步数

    # 特殊词元
    del_token = "[DEL]"
    pad_token = "<PAD>"

    # 预训练配置
    pretraining_corpus = "path/to/corpus.txt"  # 预训练语料库路径