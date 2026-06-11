"""
Tests for Check B — H-tag Hierarchy Checker.
"""
import pytest
from app.services.aeo_checks.htag_hierarchy import HtagHierarchyCheck

check = HtagHierarchyCheck()


# ── Passing case ──────────────────────────────────────────────────────────────

def test_valid_hierarchy_scores_20():
    """Clean H1 → H2 → H3 structure with no violations → 20 pts."""
    html = """
    <html><body>
      <h1>Main Title</h1>
      <h2>Section One</h2>
      <h3>Subsection</h3>
      <h2>Section Two</h2>
    </body></html>
    """
    result = check.run(html)
    assert result.score == 20
    assert result.passed is True
    assert result.details["violations"] == []
    assert result.details["h_tags_found"] == ["h1", "h2", "h3", "h2"]


# ── Missing H1 ────────────────────────────────────────────────────────────────

def test_missing_h1_scores_0():
    """No H1 present → 0 pts."""
    html = """
    <html><body>
      <h2>Section One</h2>
      <h3>Subsection</h3>
    </body></html>
    """
    result = check.run(html)
    assert result.score == 0
    assert result.passed is False
    assert any("Missing H1" in v for v in result.details["violations"])


# ── Multiple H1s ──────────────────────────────────────────────────────────────

def test_multiple_h1_scores_12():
    """Two H1 tags → 1 violation → 12 pts."""
    html = """
    <html><body>
      <h1>Title One</h1>
      <h1>Title Two</h1>
      <h2>Section</h2>
    </body></html>
    """
    result = check.run(html)
    assert result.score == 12
    assert any("Multiple H1" in v for v in result.details["violations"])


# ── Skipped level ─────────────────────────────────────────────────────────────

def test_skipped_heading_level_scores_12():
    """H1 → H3 without H2 → 1 skipped-level violation → 12 pts."""
    html = """
    <html><body>
      <h1>Title</h1>
      <h3>Jumped to H3</h3>
    </body></html>
    """
    result = check.run(html)
    assert result.score == 12
    assert any("Skipped" in v for v in result.details["violations"])


# ── Heading before H1 ─────────────────────────────────────────────────────────

def test_heading_before_h1_is_violation():
    """H2 before H1 should be flagged as a violation."""
    html = """
    <html><body>
      <h2>Intro note</h2>
      <h1>Main Title</h1>
      <h2>Section</h2>
    </body></html>
    """
    result = check.run(html)
    assert any("before H1" in v for v in result.details["violations"])


# ── 3+ violations scores 0 ────────────────────────────────────────────────────

def test_three_violations_scores_0():
    """Three or more violations → 0 pts."""
    html = """
    <html><body>
      <h2>Before H1</h2>
      <h1>Title</h1>
      <h1>Second H1</h1>
      <h4>Massive jump</h4>
    </body></html>
    """
    result = check.run(html)
    assert result.score == 0
