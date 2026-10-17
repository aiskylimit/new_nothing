"""Answer-only SFT target: the ground-truth solution, with no model-generated reasoning."""

from sgl.data.prepare import answer_only_response, reconcile_thinking_markers

S1K_ROW = {
    "solution": "128",
    "deepseek_thinking_trajectory": "first step\nsecond step",
    "deepseek_attempt": "We get x=128. The answer is \\boxed{128}.",
}
BOXED = "The final answer is \\boxed{128}."


def test_s1k_row_uses_ground_truth_not_deepseek():
    response = answer_only_response(S1K_ROW)
    assert "first step" not in response
    assert "x=128" not in response
    assert response.endswith(BOXED)


def test_non_thinking_templates_get_the_bare_solution():
    response = answer_only_response(S1K_ROW)
    qwen25 = reconcile_thinking_markers("<|im_start|>assistant\n", response)
    qwen3 = reconcile_thinking_markers("...<think>\n\n</think>\n\n", response)
    assert qwen25 == qwen3 == BOXED


def test_open_think_prompt_closes_the_block_immediately():
    # R1-Distill opens <think>; the answer-only target must close it before the solution so the
    # model is never taught to reason inside the block.
    result = reconcile_thinking_markers("...<|Assistant|><think>\n", answer_only_response(S1K_ROW))
    assert result.startswith("</think>")
    assert result.endswith(BOXED)


def test_boxed_reference_solution_is_kept_verbatim():
    solution = "Expand to get $x^2 = 4$.\nHence $x = 2$, so the answer is $\\boxed{2}$."
    assert answer_only_response({"solution": solution}).endswith(solution)


def test_long_unboxed_solution_gets_answer_appended_when_known():
    solution = "Expand to get $x^2 = 4$.\nHence $x = 2$."
    result = answer_only_response({"solution": solution, "answer": "2"})
    assert result.endswith(f"{solution}\n\nThe final answer is \\boxed{{2}}.")


def test_long_unboxed_solution_without_answer_is_kept_as_is():
    solution = "Expand to get $x^2 = 4$.\nHence $x = 2$."
    assert answer_only_response({"solution": solution}).endswith(solution)


def test_limo_row_boxes_the_bare_answer():
    assert answer_only_response({"answer": "17"}).endswith("The final answer is \\boxed{17}.")


def test_row_without_ground_truth_is_skipped():
    assert answer_only_response({"deepseek_attempt": "The answer is \\boxed{2}."}) is None
    assert answer_only_response({"solution": "   "}) is None
