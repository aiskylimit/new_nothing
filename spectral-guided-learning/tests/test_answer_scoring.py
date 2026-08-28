from answer_scoring import extract_boxed, extract_choice, normalize_math, score_generation


def test_extract_last_boxed_answer():
    text = r"First \boxed{12}, but on reflection \boxed{42}."
    assert extract_boxed(text) == "42"


def test_extract_boxed_handles_nested_braces():
    assert extract_boxed(r"so \boxed{\frac{1}{2}}") == r"\frac{1}{2}"


def test_extract_boxed_returns_none_when_absent():
    assert extract_boxed("no final answer here") is None


def test_normalize_strips_latex_noise():
    assert normalize_math(r"\left( 5 \right)") == normalize_math("(5)")
    assert normalize_math("108^\\circ".replace("^\\circ", "")) == "108"
    assert normalize_math("3.0") == "3"


def test_scores_exact_numeric_match():
    assert score_generation(r"answer is \boxed{204}", "204", "math")


def test_scores_mismatch_as_incorrect():
    assert not score_generation(r"\boxed{205}", "204", "math")


def test_missing_answer_is_incorrect_not_crash():
    assert not score_generation("I ran out of tokens while thinking", "204", "math")


def test_whitespace_and_trailing_period_ignored():
    assert score_generation(r"\boxed{ 42 }.", "42", "math")


def test_mcq_letter_from_boxed():
    assert score_generation(r"therefore \boxed{C}", "C", "mcq")


def test_mcq_letter_from_prose_fallback():
    assert extract_choice("After analysis, the answer is B") == "B"


def test_mcq_wrong_letter_rejected():
    assert not score_generation(r"\boxed{A}", "D", "mcq")


def test_mcq_uses_final_answer_not_first_mention():
    text = "Option A looks plausible... but actually the answer is D"
    assert extract_choice(text) == "D"
