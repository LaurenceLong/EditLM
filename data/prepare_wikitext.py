import os
from datasets import load_dataset

def prepare_wikitext(dataset_name, config_name, split, output_dir, output_filename):
    """
    下载指定数据集的特定部分，并将其保存为单个原始文本文件。

    Args:
        dataset_name (str): 数据集名称 (例如 'wikitext')
        config_name (str): 数据集配置名称 (例如 'wikitext-2-raw-v1')
        split (str): 要下载的数据集部分 (例如 'train', 'validation', 'test')
        output_dir (str): 保存输出文件的目录
        output_filename (str): 输出文本文件的名称
    """
    print(f"正在加载数据集: {dataset_name}, 配置: {config_name}, 部分: {split}...")
    # 加载数据集的指定部分
    # 使用 trust_remote_code=True (如果 datasets 版本需要)
    try:
        dataset = load_dataset(dataset_name, config_name, split=split, trust_remote_code=True)
    except TypeError: # Older versions might not have trust_remote_code
         dataset = load_dataset(dataset_name, config_name, split=split)


    print("数据集加载完成.")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    print(f"正在将数据写入: {output_path}")

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for example in dataset:
            text = example['text'].strip() # 获取文本并去除首尾空白
            # WikiText 有些行是空的或者只是段落标题 (例如 ' = Section Title = ')
            # 我们可以选择性地过滤掉这些，或者保留它们，取决于预训练目标
            # 这里我们保留非空行
            if text:
                f.write(text + "\n")
                count += 1

    print(f"处理完成。共写入 {count} 行非空文本到 {output_path}")

if __name__ == "__main__":
    # --- 配置 ---
    DATASET_NAME = 'wikitext'
    CONFIG_NAME = 'wikitext-2-raw-v1' # 使用原始版本，更接近纯文本
    SPLIT = 'train'                   # 预训练通常使用训练集
    OUTPUT_DIR = './data/wikitext-2-raw' # 定义保存数据的目录
    OUTPUT_FILENAME = f'{CONFIG_NAME}-{SPLIT}.txt' # 定义输出文件名
    # -------------

    prepare_wikitext(DATASET_NAME, CONFIG_NAME, SPLIT, OUTPUT_DIR, OUTPUT_FILENAME)

    # 你也可以选择性地为验证集或测试集生成文件
    # prepare_wikitext(DATASET_NAME, CONFIG_NAME, 'validation', OUTPUT_DIR, f'{CONFIG_NAME}-validation.txt')
    # prepare_wikitext(DATASET_NAME, CONFIG_NAME, 'test', OUTPUT_DIR, f'{CONFIG_NAME}-test.txt')