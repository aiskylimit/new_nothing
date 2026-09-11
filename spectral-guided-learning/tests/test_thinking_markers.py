from data_prep import reconcile_thinking_markers

S1K_RESPONSE = "<think>\nfirst step\nsecond step\n</think>\n\nThe answer is \\boxed{42}."


def test_open_think_prompt_drops_duplicate_opener():
    # R1-Distill's template opens <think> and expects the model to continue inside it.
    result = reconcile_thinking_markers("...<|Assistant|><think>\n", S1K_RESPONSE)
    assert result.startswith("first step")
    assert result.count("<think>") == 0
    assert result.count("</think>") == 1


def test_closed_empty_think_prompt_strips_response_markers():
    # Qwen3 with enable_thinking=False emits a closed empty block; a second one would nest.
    result = reconcile_thinking_markers("...<think>\n\n</think>\n\n", S1K_RESPONSE)
    assert "<think>" not in result
    assert "</think>" not in result
    assert result.startswith("first step")


def test_prompt_without_thinking_strips_response_markers():
    # Qwen2.5-Instruct has no thinking concept, so the response must reason in plain prose --
    # the same target format the Qwen3 track gets, since the comparison treats them alike.
    result = reconcile_thinking_markers("<|im_start|>assistant\n", S1K_RESPONSE)
    assert "<think>" not in result
    assert "</think>" not in result
    assert result.startswith("first step")


def test_both_non_thinking_templates_agree_on_target_format():
    qwen25 = reconcile_thinking_markers("<|im_start|>assistant\n", S1K_RESPONSE)
    qwen3 = reconcile_thinking_markers("...<think>\n\n</think>\n\n", S1K_RESPONSE)
    assert qwen25 == qwen3


def test_open_think_prompt_with_unwrapped_response_is_untouched():
    response = "first step\n</think>\n\nThe answer is \\boxed{42}."
    assert reconcile_thinking_markers("...<think>\n", response) == response
