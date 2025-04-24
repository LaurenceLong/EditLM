from transformers import AutoTokenizer


def get_tokenizer(base, **kwargs):
    # 在创建tokenizer后将pad_token作为delete_token_id标记
    tokenizer = AutoTokenizer.from_pretrained(base, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 获取 <|delete_token_id|>标记的ID
    del_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    print(f" <|delete_token_id|>标记的ID为: {del_token_id}")
    return tokenizer


def get_del_token_id(tokenizer):
    return tokenizer.pad_token
