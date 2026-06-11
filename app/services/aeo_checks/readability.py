from __future__ import annotations
import re
import textstat
from app.models.schemas import CheckResult
from app.services.aeo_checks.base import BaseCheck
from app.services.content_parser import get_plain_text


def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter on . ! ? boundaries."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.split()) >= 4]


def _syllable_density(sentence: str) -> float:
    """Syllables per word — used to rank complexity."""
    words = sentence.split()
    if not words:
        return 0.0
    syllables = textstat.syllable_count(sentence)
    return syllables / len(words)


def _top_complex_sentences(text: str, n: int = 3) -> list[str]:
    sentences = _split_sentences(text)
    ranked = sorted(sentences, key=_syllable_density, reverse=True)
    return ranked[:n]


def _compute_score(grade: float) -> int:
    if 7 <= grade <= 9:
        return 20
    if grade in (6, 10) or 6 <= grade < 7 or 9 < grade <= 10:
        return 14
    if grade in (5, 11) or 5 <= grade < 6 or 10 < grade <= 11:
        return 8
    return 0  # ≤ 4 or ≥ 12


def _recommendation(grade: float) -> str | None:
    if 7 <= grade <= 9:
        return None
    if grade > 9:
        return (
            f"Content reads at Grade {grade:.1f}. Shorten sentences and replace "
            "technical jargon with plain language to reach Grade 7–9."
        )
    return (
        f"Content reads at Grade {grade:.1f}. Add more specificity and "
        "structured sentences to reach Grade 7–9."
    )


class ReadabilityCheck(BaseCheck):
    check_id = "readability"
    name = "Snippet Readability"
    max_score = 20

    def run(self, html: str) -> CheckResult:
        plain = get_plain_text(html)
        grade = textstat.flesch_kincaid_grade(plain)
        complex_sentences = _top_complex_sentences(plain)
        score = _compute_score(grade)
        rec = _recommendation(grade)

        return CheckResult(
            check_id=self.check_id,
            name=self.name,
            passed=score == self.max_score,
            score=score,
            max_score=self.max_score,
            details={
                "fk_grade_level": round(grade, 1),
                "target_range": "7-9",
                "complex_sentences": complex_sentences,
            },
            recommendation=rec,
        )
