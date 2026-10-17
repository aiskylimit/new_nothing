"""Model-specific handling for generation prompts in the offline workflow."""


def apply_generation_prompt(tokenizer, messages, enable_thinking=True):
    # Olmo-3-7B-Think opens <think> in its chat template but ignores Qwen's
    # enable_thinking argument. Close the empty block for non-thinking rollouts.
    if "olmo-3" in tokenizer.name_or_path.lower():
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if not enable_thinking:
            if not text.rstrip().endswith("<think>"):
                raise ValueError("Expected the OLMo-3 chat template to end with an open <think> tag")
            text += "\n\n</think>\n\n"
        return text
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
