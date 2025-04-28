import argparse
from transformers import pipeline


def main():
    parser = argparse.ArgumentParser(
        description="使用命令行启动文本生成任务，可设置输入内容和使用的模型。"
    )
    parser.add_argument(
        "--content",
        type=str,
        default="Fix this calculation please: 12 + 13 = 42",
        help="待处理的文本内容"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="用于文本生成的模型名称"
    )
    args = parser.parse_args()

    # 初始化文本生成管道
    pipe = pipeline("text-generation", model=args.model)

    # 构造消息（包含角色和内容，可以按需扩展）
    messages = [{"role": "user", "content": args.content}]

    # 调用管道生成文本
    outputs = pipe(messages)
    print(outputs)


if __name__ == "__main__":
    main()
