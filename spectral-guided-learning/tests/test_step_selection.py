import pytest

from step_selection import build_loss_mask, select_steps_by_energy, selection_stats


def test_selects_minimal_set_reaching_threshold():
    # total = 10; sorted desc: 5, 3, 1.5, 0.5 -> p=0.8 needs 5+3 = 8 >= 8
    strengths = [1.5, 5.0, 0.5, 3.0]
    assert select_steps_by_energy(strengths, 0.8) == [1, 3]


def test_higher_threshold_selects_superset():
    strengths = [1.5, 5.0, 0.5, 3.0]
    low = set(select_steps_by_energy(strengths, 0.7))
    high = set(select_steps_by_energy(strengths, 0.95))
    assert low.issubset(high)


def test_always_keeps_at_least_one_step():
    assert select_steps_by_energy([1.0, 1.0, 1.0], 0.0) == [0]


def test_threshold_one_keeps_all_steps():
    assert select_steps_by_energy([2.0, 1.0, 3.0], 1.0) == [0, 1, 2]


def test_ties_broken_by_index_deterministically():
    for _ in range(5):
        assert select_steps_by_energy([1.0, 1.0, 1.0, 1.0], 0.5) == [0, 1]


def test_zero_total_strength_keeps_everything():
    assert select_steps_by_energy([0.0, 0.0], 0.8) == [0, 1]


def test_empty_input_returns_empty():
    assert select_steps_by_energy([], 0.8) == []


def test_build_loss_mask_marks_only_selected_spans():
    mask = build_loss_mask(10, [(2, 5), (5, 8), (8, 10)], selected_steps=[0, 2])
    assert mask == [0, 0, 1, 1, 1, 0, 0, 0, 1, 1]


def test_prompt_positions_are_never_supervised():
    mask = build_loss_mask(6, [(3, 6)], selected_steps=[0])
    assert mask[:3] == [0, 0, 0]


def test_selection_stats_reports_retention():
    stats = selection_stats([(0, 10), (10, 20), (20, 30)], selected_steps=[0, 2])
    assert stats["step_retention"] == 2 / 3
    assert stats["token_retention"] == 20 / 30
    assert stats["keeps_final_step"] == 1.0


def test_selection_stats_reports_step_and_token_drops_separately():
    # steps of unequal length: 1 of 3 steps dropped, but only 5 of 25 tokens
    stats = selection_stats([(0, 10), (10, 15), (15, 25)], selected_steps=[0, 2])
    assert stats["n_steps_dropped"] == 1
    assert stats["step_drop"] == pytest.approx(1 / 3)
    assert stats["n_tokens_dropped"] == 5
    assert stats["token_drop"] == pytest.approx(5 / 25)


def test_drop_and_retention_are_complementary():
    stats = selection_stats([(0, 4), (4, 8), (8, 12), (12, 16)], selected_steps=[1])
    assert stats["step_drop"] + stats["step_retention"] == 1.0
    assert stats["token_drop"] + stats["token_retention"] == 1.0
