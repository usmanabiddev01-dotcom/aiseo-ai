from __future__ import annotations
import spacy
from app.models.schemas import CheckResult
from app.services.aeo_checks.base import BaseCheck
from app.services.content_parser import extract_first_paragraph

HEDGE_PHRASES = [
    "it depends",
    "may vary",
    "in some cases",
    "this varies",
    "generally speaking",
]

# Lazy-load spaCy model once per process
_nlp = None


def _get_nlp() -> spacy.language.Language:
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
    return _nlp


def _is_declarative(text: str) -> bool:
    """Return True if the text contains a subject + root verb (declarative sentence)."""
    nlp = _get_nlp()
    doc = nlp(text)
    has_subject = any(tok.dep_ in ("nsubj", "nsubjpass") for tok in doc)
    has_root_verb = any(tok.dep_ == "ROOT" and tok.pos_ == "VERB" for tok in doc)
    return has_subject and has_root_verb


def _has_hedge(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in HEDGE_PHRASES)


def _compute_score(word_count: int, is_declarative: bool, has_hedge: bool) -> int:
    if word_count > 90:
        return 0
    if word_count > 60:
        return 8
    # ≤ 60 words
    if is_declarative and not has_hedge:
        return 20
    return 12  # ≤ 60 but hedging or incomplete


def _recommendation(word_count: int, is_declarative: bool, has_hedge: bool) -> str | None:
    if word_count > 90:
        return (
            f"Your opening paragraph is {word_count} words. "
            "Trim it to under 60 words with a direct, declarative answer."
        )
    if word_count > 60:
        return (
            f"Your opening paragraph is {word_count} words. "
            "Trim it to under 60 words."
        )
    if has_hedge:
        return (
            "Remove hedge phrases (e.g. 'it depends', 'may vary') from your "
            "opening paragraph to give a confident direct answer."
        )
    if not is_declarative:
        return (
            "Rewrite your opening paragraph as a complete declarative statement "
            "with a clear subject and verb."
        )
    return None


class DirectAnswerCheck(BaseCheck):
    check_id = "direct_answer"
    name = "Direct Answer Detection"
    max_score = 20

    def run(self, html: str) -> CheckResult:
        paragraph = extract_first_paragraph(html)
        word_count = len(paragraph.split()) if paragraph else 0
        is_declarative = _is_declarative(paragraph) if paragraph else False
        has_hedge = _has_hedge(paragraph) if paragraph else False
        score = _compute_score(word_count, is_declarative, has_hedge)
        rec = _recommendation(word_count, is_declarative, has_hedge)

        return CheckResult(
            check_id=self.check_id,
            name=self.name,
            passed=score == self.max_score,
            score=score,
            max_score=self.max_score,
            details={
                "word_count": word_count,
                "threshold": 60,
                "is_declarative": is_declarative,
                "has_hedge_phrase": has_hedge,
            },
            recommendation=rec,
        )
