"""Tests for the LLM Optimization Engine's accept/reject logic.

Owner: Bryson Wingate (LLM Optimization Engine).

These test classify_candidate() directly with plain floats — no GPU,
no Ollama, no real benchmark run required. The goal is to pin down the
noise-margin boundary deterministically, since real benchmark numbers
won't reliably land in the narrow "faster but not by enough" band on
any given run.
"""

from ai_optimizer import NOISE_MARGIN, classify_candidate


def test_default_margin_is_two_percent():
    assert NOISE_MARGIN == 0.02


def test_clear_improvement_is_accepted():
    # 10% faster than best: comfortably past the 2% margin.
    assert classify_candidate(0.90, 1.00) == "improved"


def test_exactly_at_margin_is_not_improved():
    # Exactly 2% faster == the threshold itself; strict "<" means this
    # boundary value does NOT count as improved.
    assert classify_candidate(0.98, 1.00) == "within_margin"


def test_just_inside_margin_is_improved():
    # Just over 2% faster: should cross the strict "<" threshold.
    assert classify_candidate(0.979, 1.00) == "improved"


def test_faster_but_within_margin_is_flagged_not_accepted():
    # 1% faster than best: numerically better, but inside the 2% margin,
    # so it should NOT be accepted as an improvement.
    assert classify_candidate(0.99, 1.00) == "within_margin"


def test_slower_candidate_is_no_improvement():
    assert classify_candidate(1.05, 1.00) == "no_improvement"


def test_equal_candidate_is_no_improvement():
    assert classify_candidate(1.00, 1.00) == "no_improvement"


def test_custom_margin_is_respected():
    # With a 10% margin, a 5% improvement should NOT count.
    assert classify_candidate(0.95, 1.00, margin=0.10) == "within_margin"
    # But a 15% improvement should.
    assert classify_candidate(0.85, 1.00, margin=0.10) == "improved"
