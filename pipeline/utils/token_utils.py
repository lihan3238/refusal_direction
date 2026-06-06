def decode_token_labels(tokenizer, token_ids):
    return [tokenizer.decode([token_id]) for token_id in token_ids]
