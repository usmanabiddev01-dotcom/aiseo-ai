from __future__ import annotations
from fastapi import APIRouter, HTTPException
from app.models.schemas import AEORequest, AEOResponse
from app.services.content_parser import fetch_content
from app.services.aeo_checks.direct_answer import DirectAnswerCheck
from app.services.aeo_checks.htag_hierarchy import HtagHierarchyCheck
from app.services.aeo_checks.readability import ReadabilityCheck

router = APIRouter()

CHECKS = [DirectAnswerCheck(), HtagHierarchyCheck(), ReadabilityCheck()]
MAX_RAW = sum(c.max_score for c in CHECKS)  # 60

BANDS = [
    (85, "AEO Optimized"),
    (65, "Needs Improvement"),
    (40, "Significant Gaps"),
    (0,  "Not AEO Ready"),
]


def _band(score: float) -> str:
    for threshold, label in BANDS:
        if score >= threshold:
            return label
    return "Not AEO Ready"


@router.post("/analyze", response_model=AEOResponse)
async def analyze(request: AEORequest):
    try:
        html = fetch_content(request.input_type, request.input_value)
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "url_fetch_failed",
                "message": "Could not retrieve content from the provided URL.",
                "detail": str(e),
            },
        )

    results = [check.run(html) for check in CHECKS]
    raw_score = sum(r.score for r in results)
    aeo_score = round((raw_score / MAX_RAW) * 100, 1)

    return AEOResponse(
        aeo_score=aeo_score,
        band=_band(aeo_score),
        checks=results,
    )
