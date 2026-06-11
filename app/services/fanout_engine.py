from __future__ import annotations
import json
import os
import re
import time
from google import genai
from google.genai import types as genai_types
from app.models.schemas import SubQuery

MODEL_NAME = "gemini-1.5-flash"
MAX_RETRIES = 3
VALID_TYPES = {
    "comparative",
    "feature_specific",
    "use_case",
    "trust_signals",
    "how_to",
    "definitional",
}

SYSTEM_PROMPT = """\
You are a search intelligence engine. Your job is to decompose a user's target query into \
sub-queries that simulate how AI search engines (like Perplexity, ChatGPT Search, and Google AI Mode) \
build a comprehensive answer by gathering information from multiple angles.

## Task
Given a target query, generate between 10 and 15 sub-queries spread across exactly these 6 types:

1. comparative      — Compare the subject against alternatives or competitors
2. feature_specific — Focus on a specific capability, feature, or attribute
3. use_case         — Real-world application or scenario-based query
4. trust_signals    — Reviews, case studies, credibility, social proof
5. how_to           — Procedural or instructional query
6. definitional     — Conceptual "what is" or "what does X mean" query

## Rules
- Generate AT LEAST 2 sub-queries for EACH of the 6 types above (minimum 12 total)
- Generate no more than 15 sub-queries total
- Each sub-query must be a realistic, natural search query a person would type
- Sub-queries must be topically grounded in the target query — do not invent unrelated topics
- Return ONLY a valid JSON object. No markdown, no code fences, no prose. Raw JSON only.
- Do not add any fields beyond "sub_queries", "type", and "query"

## Output Format
Return exactly this JSON structure:

{
  "sub_queries": [
    {"type": "comparative", "query": "..."},
    {"type": "feature_specific", "query": "..."},
    {"type": "use_case", "query": "..."},
    {"type": "trust_signals", "query": "..."},
    {"type": "how_to", "query": "..."},
    {"type": "definitional", "query": "..."}
  ]
}

## Example (for target query: "best project management software for remote teams")

{
  "sub_queries": [
    {"type": "comparative", "query": "Asana vs Monday vs Notion for remote team project management"},
    {"type": "comparative", "query": "Jira vs Linear for distributed software teams"},
    {"type": "feature_specific", "query": "project management software with async video updates for remote teams"},
    {"type": "feature_specific", "query": "PM tool with time zone overlap visualization"},
    {"type": "use_case", "query": "project management software for fully remote startup with 10 engineers"},
    {"type": "use_case", "query": "best PM tool for agency managing multiple client projects remotely"},
    {"type": "trust_signals", "query": "remote team project management software reviews G2 2025"},
    {"type": "trust_signals", "query": "case studies companies that improved remote productivity with PM software"},
    {"type": "how_to", "query": "how to manage remote team projects without daily standups"},
    {"type": "how_to", "query": "how to set up async sprints in project management software"},
    {"type": "definitional", "query": "what is asynchronous project management"},
    {"type": "definitional", "query": "what makes project management software remote-first"}
  ]
}
"""


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences that some LLMs wrap JSON in."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_and_validate(raw: str) -> list[SubQuery]:
    """Parse LLM response into a validated list of SubQuery objects."""
    cleaned = _strip_code_fences(raw)
    data = json.loads(cleaned)

    if "sub_queries" not in data:
        raise ValueError("Response missing 'sub_queries' key")

    items = data["sub_queries"]
    if not isinstance(items, list) or len(items) < 10:
        raise ValueError(f"Expected ≥10 sub-queries, got {len(items)}")

    validated: list[SubQuery] = []
    for item in items[:15]:  # cap at 15
        if item.get("type") not in VALID_TYPES:
            raise ValueError(f"Unknown type: {item.get('type')}")
        validated.append(SubQuery(type=item["type"], query=item["query"]))

    # Verify minimum coverage: at least 1 of each type present
    found_types = {sq.type for sq in validated}
    missing = VALID_TYPES - found_types
    if missing:
        raise ValueError(f"Missing sub-query types: {missing}")

    return validated


# ---------------------------------------------------------------------------
# MOCK MODE
# Set MOCK_LLM=true in your environment (or .env) to bypass the real Gemini
# call and return a fixed fixture response. Use this when hitting free-tier
# rate limits during local development. Do NOT enable in production.
# ---------------------------------------------------------------------------
_MOCK_RESPONSE = json.dumps({
    "sub_queries": [
        {"type": "comparative",      "query": "Jasper AI vs Copy.ai vs Surfer SEO for content optimization"},
        {"type": "comparative",      "query": "AI writing tool vs human writers for SEO content quality"},
        {"type": "feature_specific", "query": "AI writing tool with real-time SERP analysis and keyword suggestions"},
        {"type": "feature_specific", "query": "AI SEO content tool with plagiarism checker and readability scoring"},
        {"type": "use_case",         "query": "AI writing tool for agency managing 50 client blogs per month"},
        {"type": "use_case",         "query": "best AI writer for e-commerce product descriptions at scale"},
        {"type": "trust_signals",    "query": "AI SEO writing tool case studies and ROI data 2025"},
        {"type": "trust_signals",    "query": "Jasper AI reviews from marketing professionals G2 Capterra"},
        {"type": "how_to",           "query": "how to use AI writing tools to optimize blog content for SEO"},
        {"type": "how_to",           "query": "how to prompt AI to write content that ranks in Google"},
        {"type": "definitional",     "query": "what is AI-assisted SEO content writing"},
        {"type": "definitional",     "query": "what makes AI-generated content rank in search engines"},
    ]
})


def generate_sub_queries(target_query: str) -> list[SubQuery]:
    """
    Call Gemini to generate sub-queries. Retries up to MAX_RETRIES times
    with exponential backoff on JSON parse or validation failures.
    Raises RuntimeError after all retries are exhausted.

    Set MOCK_LLM=true to skip the real API call and return a cached fixture
    response (useful when hitting free-tier rate limits during development).
    """
    if os.environ.get("MOCK_LLM", "").lower() == "true":
        return _parse_and_validate(_MOCK_RESPONSE)

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

    client = genai.Client(api_key=api_key)
    user_prompt = f'Target query: "{target_query}"\n\nGenerate the sub-queries now. Return raw JSON only.'

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=user_prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                ),
            )
            sub_queries = _parse_and_validate(response.text)
            return sub_queries
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)  # 2s, 4s backoff
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"JSONDecodeError on attempt {MAX_RETRIES}: {last_error}"
    )
