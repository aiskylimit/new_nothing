from sgl.eval.evaluate import palign_close_thinking

R1_PROMPT = "<｜begin▁of▁sentence｜><｜User｜>q<｜Assistant｜><think>\n"


def test_matches_palign_test_py_for_r1_distill():
    # P-ALIGN/src/test.py apply_chat(): strip the open <think>, append "<think>\n\n</think>\n\n"
    assert palign_close_thinking(R1_PROMPT) == (
        "<｜begin▁of▁sentence｜><｜User｜>q<｜Assistant｜><think>\n\n</think>\n\n"
    )


def test_leaves_closed_or_thinkless_prompts_alone():
    for prompt in ("<|im_start|>assistant\n<think>\n\n</think>\n\n", "<|im_start|>assistant\n"):
        assert palign_close_thinking(prompt) == prompt
