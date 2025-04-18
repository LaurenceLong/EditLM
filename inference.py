"""
推理脚本，用于使用训练好的模型生成编辑
"""
import argparse
import torch
from model.modeling import EditLM
from model.tokenizer import get_tokenizer
from utils.edit_utils import generate_edits
from config.config import Config

def main():
    parser = argparse.ArgumentParser(description="Inference script for Index-Token Editor model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--source_text", type=str, required=True, help="Source text to edit")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    args = parser.parse_args()

    # 加载配置
    config = Config()

    # 加载词元器
    tokenizer = get_tokenizer(config)

    # 初始化模型
    model = EditLM(
        config=config,
        model_name_or_path=config.model_name_or_path,
        tokenizer_length=len(tokenizer)
    )

    # 加载训练好的权重
    model.load_state_dict(torch.load(f"{args.model_path}/model.pt", map_location=args.device))
    model.to(args.device)
    model.eval()

    # 生成编辑
    edited_text = generate_edits(model, tokenizer, args.source_text, config, args.device)

    print("Source text:", args.source_text)
    print("Edited text:", edited_text)

if __name__ == "__main__":
    main()
