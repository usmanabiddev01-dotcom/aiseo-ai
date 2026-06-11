from __future__ import annotations
from bs4 import BeautifulSoup
from app.models.schemas import CheckResult
from app.services.aeo_checks.base import BaseCheck

HEADING_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6"]


def _parse_htags(html: str) -> list[str]:
    """Return ordered list of heading tag names found in the HTML."""
    soup = BeautifulSoup(html, "html.parser")
    return [tag.name for tag in soup.find_all(HEADING_TAGS)]


def _find_violations(htags: list[str]) -> list[str]:
    violations: list[str] = []

    h1_count = htags.count("h1")
    if h1_count == 0:
        violations.append("Missing H1: no <h1> tag found in the document.")
        return violations  # remaining checks are meaningless without H1
    if h1_count > 1:
        violations.append(f"Multiple H1s: found {h1_count} <h1> tags; exactly one is required.")

    # Check nothing appears before the first H1
    first_h1_index = htags.index("h1")
    if first_h1_index > 0:
        preceding = htags[:first_h1_index]
        violations.append(
            f"Heading(s) before H1: {preceding} appear before the <h1>."
        )

    # Check no level is skipped (e.g. H1 → H3 without H2)
    level = lambda tag: int(tag[1])  # "h2" → 2
    prev_level = 1  # start after H1
    for tag in htags[first_h1_index + 1 :]:
        current = level(tag)
        if current > prev_level + 1:
            violations.append(
                f"Skipped heading level: jumped from H{prev_level} to H{current}."
            )
        prev_level = current

    return violations


def _compute_score(violations: list[str], htags: list[str]) -> int:
    if "h1" not in htags:
        return 0
    n = len(violations)
    if n == 0:
        return 20
    if n <= 2:
        return 12
    return 0


def _recommendation(violations: list[str]) -> str | None:
    if not violations:
        return None
    return "Fix heading structure: " + " | ".join(violations)


class HtagHierarchyCheck(BaseCheck):
    check_id = "htag_hierarchy"
    name = "H-tag Hierarchy"
    max_score = 20

    def run(self, html: str) -> CheckResult:
        htags = _parse_htags(html)
        violations = _find_violations(htags)
        score = _compute_score(violations, htags)
        rec = _recommendation(violations)

        return CheckResult(
            check_id=self.check_id,
            name=self.name,
            passed=score == self.max_score,
            score=score,
            max_score=self.max_score,
            details={
                "violations": violations,
                "h_tags_found": htags,
            },
            recommendation=rec,
        )
