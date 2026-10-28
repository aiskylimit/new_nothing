"""Chat-format tokens of the instruction models we train: Qwen3, Llama-3 and Gemma-3.

Each training target ends with the token that closes an assistant turn, and generation stops
on that token or on the end-of-text token. Both are read from the tokenizer's chat template,
so the code holds no model-specific token ids.
"""

# --model-type values whose tokenized data is uint32 with the -1 prompt/response sentinel
CHAT_MODEL_TYPES = ("qwen", "llama", "gemma")

# end of an assistant turn -> end of text, per chat template
TURN_END = {
    "<|im_end|>": "<|endoftext|>",      # Qwen
    "<|eot_id|>": "<|end_of_text|>",    # Llama 3
    "<end_of_turn>": "<eos>",           # Gemma 3
}

# Extra apply_chat_template variables. enable_thinking is Qwen3's. Llama 3 writes today's date
# into the system turn unless date_string is given, so the same row would tokenize differently
# from one day to the next. A template ignores variables it does not use.
TEMPLATE_KWARGS = {"enable_thinking": False, "date_string": "26 Jul 2024"}


def stop_ids(tokenizer):
    """(end-of-turn id, end-of-text id) of the tokenizer's chat format."""
    template = tokenizer.chat_template or ""
    for turn_end, text_end in TURN_END.items():
        if turn_end in template:
            return tokenizer.convert_tokens_to_ids(turn_end), tokenizer.convert_tokens_to_ids(text_end)
    raise ValueError(f"no known end-of-turn token in the chat template of {tokenizer.name_or_path}")
