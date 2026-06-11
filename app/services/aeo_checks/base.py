from __future__ import annotations
from abc import ABC, abstractmethod
from app.models.schemas import CheckResult


class BaseCheck(ABC):
    check_id: str
    name: str
    max_score: int

    @abstractmethod
    def run(self, html: str) -> CheckResult:
        """Run the check against the provided HTML/text and return a CheckResult."""
        ...
