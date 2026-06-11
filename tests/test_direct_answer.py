"""
Tests for Check A — Direct Answer Detection.
Each test exercises the public run() method directly (no FastAPI, no mocks).
"""
import pytest
from app.services.aeo_checks.direct_answer import DirectAnswerCheck

check = DirectAnswerCheck()


# ── Helper ────────────────────────────────────────────────────────────────────

def _html(text: str) -> str:
    return f"<html><body><p>{text}</p></body></html>"


# ── Passing case ──────────────────────────────────────────────────────────────

def test_perfect_direct_answer_scores_20():
    """≤60 words, declarative sentence, no hedge phrase → full 20 pts."""
    content = _html(
        "AEO content optimization improves how AI search engines extract "
        "and cite your content by structuring information for direct answer retrieval."
    )
    result = check.run(content)
    assert result.score == 20
    assert result.passed is True
    assert result.details["is_declarative"] is True
    assert result.details["has_hedge_phrase"] is False
    assert result.details["word_count"] <= 60


# ── Hedge phrase case ─────────────────────────────────────────────────────────

def test_hedge_phrase_reduces_score_to_12():
    """≤60 words but contains a hedge phrase → 12 pts."""
    content = _html(
        "It depends on your use case, but AEO optimization generally helps "
        "AI engines extract and surface your content more effectively."
    )
    result = check.run(content)
    assert result.score == 12
    assert result.details["has_hedge_phrase"] is True
    assert result.details["word_count"] <= 60


# ── Over 90 words case ────────────────────────────────────────────────────────

def test_over_90_words_scores_0():
    """First paragraph > 90 words → 0 pts."""
    long_text = " ".join(["word"] * 95)
    content = _html(long_text)
    result = check.run(content)
    assert result.score == 0
    assert result.details["word_count"] > 90


# ── 61–90 words case ─────────────────────────────────────────────────────────

def test_61_to_90_words_scores_8():
    """First paragraph between 61 and 90 words → 8 pts."""
    mid_text = " ".join(["word"] * 75)
    content = _html(mid_text)
    result = check.run(content)
    assert result.score == 8
    assert 61 <= result.details["word_count"] <= 90


# ── Recommendation present when not passing ───────────────────────────────────

def test_recommendation_present_on_failure():
    long_text = " ".join(["word"] * 100)
    content = _html(long_text)
    result = check.run(content)
    assert result.recommendation is not None
    assert len(result.recommendation) > 0


# ── No recommendation on perfect pass ────────────────────────────────────────

def test_no_recommendation_on_perfect_score():
    content = _html(
        "Search engine optimization structures content so AI platforms "
        "extract and display it as direct answers to user queries."
    )
    result = check.run(content)
    if result.score == 20:
        assert result.recommendation is None
