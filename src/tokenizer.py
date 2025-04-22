from transformers import AutoTokenizer

NUM_ADDED_TOKENS = [0]


def get_tokenizer(base, **kwargs):
    # 在创建tokenizer后添加 <|delete_token_id|>标记
    tokenizer = AutoTokenizer.from_pretrained(base, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 添加 <|delete_token_id|>特殊标记
    special_tokens_dict = {'additional_special_tokens': [' <|delete_token_id|>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"添加了 {num_added_toks} 个特殊标记")

    # 获取 <|delete_token_id|>标记的ID
    del_token_id = tokenizer.convert_tokens_to_ids(' <|delete_token_id|>')
    print(f" <|delete_token_id|>标记的ID为: {del_token_id}")

    global NUM_ADDED_TOKENS
    NUM_ADDED_TOKENS[0] = num_added_toks
    return tokenizer


def get_del_token_id(tokenizer):
    return tokenizer.convert_tokens_to_ids(' <|delete_token_id|>')
