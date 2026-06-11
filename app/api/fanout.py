from __future__ import annotations
from fastapi import APIRouter, HTTPException
from app.models.schemas import FanOutRequest, FanOutResponse, GapSummary
from app.services.fanout_engine import generate_sub_queries, MODEL_NAME
from app.services.gap_analyzer import analyze_gaps

router = APIRouter()


@router.post("/generate", response_model=FanOutResponse)
async def generate(request: FanOutRequest):
    try:
        sub_queries = generate_sub_queries(request.target_query)
    except RuntimeError as e:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "llm_unavailable",
                "message": "Fan-out generation failed. The LLM returned an invalid response after 3 retries.",
                "detail": str(e),
            },
        )

    gap_summary = None
    if request.existing_content:
        sub_queries = analyze_gaps(sub_queries, request.existing_content)

        covered_sqs = [sq for sq in sub_queries if sq.covered]
        covered_types = list({sq.type for sq in covered_sqs})
        missing_types = list({sq.type for sq in sub_queries if not sq.covered})
        total = len(sub_queries)
        coverage_pct = round((len(covered_sqs) / total) * 100, 1) if total else 0.0

        gap_summary = GapSummary(
            covered=len(covered_sqs),
            total=total,
            coverage_percent=coverage_pct,
            covered_types=covered_types,
            missing_types=missing_types,
        )

    return FanOutResponse(
        target_query=request.target_query,
        model_used=MODEL_NAME,
        total_sub_queries=len(sub_queries),
        sub_queries=sub_queries,
        gap_summary=gap_summary,
    )
