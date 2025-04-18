"""
自定义词元器，扩展原有的Llama2词元器，添加特殊符号
"""
from transformers import AutoTokenizer, PreTrainedTokenizer
from config.config import Config

def get_tokenizer(config: Config) -> PreTrainedTokenizer:
    """加载并扩展tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

    # 添加特殊词元
    special_tokens = {
        "additional_special_tokens": [
            config.del_token,     # 删除操作符
        ]
    }

    # 若原始词元器没有pad_token，则添加
    if tokenizer.pad_token is None:
        if config.pad_token in tokenizer.get_vocab():
            tokenizer.pad_token = config.pad_token
        else:
            special_tokens["pad_token"] = config.pad_token

    # 调整词元器以包含新的特殊词元
    tokenizer.add_special_tokens(special_tokens)

    # 设置特殊词元ID的属性用于方便访问
    tokenizer.del_token_id = tokenizer.convert_tokens_to_ids(config.del_token)

    return tokenizer
