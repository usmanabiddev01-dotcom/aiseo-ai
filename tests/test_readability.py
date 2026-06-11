"""
Tests for Check C — Snippet Readability Scorer.
"""
import pytest
from app.services.aeo_checks.readability import ReadabilityCheck, _compute_score, _top_complex_sentences

check = ReadabilityCheck()


# ── Score table unit tests (pure function, no HTML needed) ────────────────────

def test_score_grade_8_returns_20():
    assert _compute_score(8.0) == 20

def test_score_grade_7_returns_20():
    assert _compute_score(7.0) == 20

def test_score_grade_9_returns_20():
    assert _compute_score(9.0) == 20

def test_score_grade_10_returns_14():
    assert _compute_score(10.0) == 14

def test_score_grade_6_returns_14():
    assert _compute_score(6.0) == 14

def test_score_grade_11_returns_8():
    assert _compute_score(11.0) == 8

def test_score_grade_5_returns_8():
    assert _compute_score(5.0) == 8

def test_score_grade_12_returns_0():
    assert _compute_score(12.0) == 0

def test_score_grade_4_returns_0():
    assert _compute_score(4.0) == 0

def test_score_grade_above_12_returns_0():
    assert _compute_score(13.5) == 0


# ── Integration: run() on real HTML ──────────────────────────────────────────

def test_simple_content_returns_check_result():
    """run() should return a CheckResult with expected fields."""
    html = """
    <html><body>
      <p>The cat sat on the mat. Dogs run fast. Birds fly high in the sky.
      Simple words make easy reading. Kids learn this in class every day.</p>
    </body></html>
    """
    result = check.run(html)
    assert result.check_id == "readability"
    assert "fk_grade_level" in result.details
    assert "target_range" in result.details
    assert "complex_sentences" in result.details
    assert isinstance(result.details["complex_sentences"], list)
    assert result.score in (0, 8, 14, 20)


def test_boilerplate_stripped_before_scoring():
    """Footer and nav content should not influence the FK grade."""
    html = """
    <html><body>
      <nav>Home About Contact Privacy Policy Terms of Service</nav>
      <p>The cat sat on the mat. Simple words help readers understand content quickly.</p>
      <footer>Copyright 2025 Company Name All Rights Reserved</footer>
    </body></html>
    """
    result = check.run(html)
    assert result.check_id == "readability"
    assert result.score in (0, 8, 14, 20)


# ── Complex sentence ranking ──────────────────────────────────────────────────

def test_top_complex_sentences_returns_up_to_3():
    text = (
        "The cat sat. "
        "Multifaceted optimization paradigms necessitate sophisticated methodological frameworks. "
        "Dogs run fast. "
        "Heuristic algorithmic decomposition of multivariable regression coefficients. "
        "Birds fly. "
        "Contemporaneous epistemological underpinnings demonstrate unprecedented complexity."
    )
    result = _top_complex_sentences(text, n=3)
    assert len(result) <= 3
    # The complex sentences should rank above simple ones
    assert any(len(s.split()) > 5 for s in result)


def test_complex_sentences_fewer_than_3_available():
    text = "The cat sat. Dogs run."
    result = _top_complex_sentences(text, n=3)
    assert len(result) <= 3
