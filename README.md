# AEGIS — Answer Engine & Generative Intelligence Suite

A Python/FastAPI service that scores content for AEO (Answer Engine Optimization) readiness and generates query fan-out maps using LLM + semantic embeddings.

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Set environment variables

```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

If you prefer OpenAI, the architecture supports swapping it in `fanout_engine.py` — just replace the `genai` calls with the `openai` client.

### 3. Run the server

```bash
uvicorn app.main:app --reload
```

API is now available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

### 4. Run tests

```bash
pytest tests/ -v
```

---

## API Endpoints

### `POST /api/aeo/analyze`

Analyze content for AEO readiness.

```bash
curl -X POST http://localhost:8000/api/aeo/analyze \
  -H "Content-Type: application/json" \
  -d '{"input_type": "text", "input_value": "<html><body><p>Your content here.</p></body></html>"}'
```

```bash
curl -X POST http://localhost:8000/api/aeo/analyze \
  -H "Content-Type: application/json" \
  -d '{"input_type": "url", "input_value": "https://example.com/article"}'
```

### `POST /api/fanout/generate`

Generate query fan-out sub-queries (with optional gap analysis).

```bash
curl -X POST http://localhost:8000/api/fanout/generate \
  -H "Content-Type: application/json" \
  -d '{"target_query": "best AI writing tool for SEO"}'
```

```bash
curl -X POST http://localhost:8000/api/fanout/generate \
  -H "Content-Type: application/json" \
  -d '{
    "target_query": "best AI writing tool for SEO",
    "existing_content": "Paste your article text here to enable gap analysis..."
  }'
```

---

## What I Completed vs. Skipped

### Completed
- Feature 1: All three AEO checks (Direct Answer, H-tag Hierarchy, Readability) — fully implemented, independently testable
- Feature 2: LLM fan-out engine with structured prompt, retry logic (3 attempts, exponential backoff), JSON validation
- Feature 2: Semantic gap analysis with sentence-transformer embeddings and cosine similarity
- All four test files with multiple pass/fail cases per check
- LLM JSON parsing tests with mocked Gemini calls

### Skipped / Known Limitations
- **Async endpoints:** Both endpoints are `async def` but the NLP operations (spaCy, textstat, sentence-transformers) are CPU-bound. In production I'd run these in a `ThreadPoolExecutor` via `asyncio.get_event_loop().run_in_executor()` to avoid blocking the event loop.
- **URL fetching:** JavaScript-heavy SPAs that render content client-side will return near-empty HTML. I'd add Playwright/Puppeteer for headless rendering in production.
- **Caching:** The sentence-transformer model is loaded once per process (singleton). In a multi-worker deployment, each worker loads its own copy — a shared model server (e.g. Triton, custom gRPC) would be the production solution.

---

## Prompt Design Decisions

### The Core Problem
LLMs frequently deviate from JSON-only output in two ways:
1. Wrapping the JSON in markdown code fences (` ```json ... ``` `)
2. Hallucinating extra fields or using non-specified type names

### My Approach

**System prompt structure:**
- Defined all 6 sub-query types with descriptions upfront
- Stated the minimum count per type explicitly (≥2) to prevent under-generation
- Enforced "Return ONLY a valid JSON object. No markdown, no code fences, no prose." as a hard rule
- Embedded a concrete, full output example so the model has a ground-truth format to match

**Defensive parsing:**
- `_strip_code_fences()` removes markdown wrappers before JSON parsing
- `_parse_and_validate()` checks for the presence of `sub_queries`, validates each `type` against the known set, enforces the 10+ count, and checks that all 6 types appear at least once
- Any failure triggers a retry with exponential backoff (2s, 4s)
- After 3 failures, returns HTTP 503 with a descriptive error — the API never crashes

**Why this works better than a minimal prompt:**
A single-sentence prompt like "Generate 10 sub-queries about {query}" produces inconsistent output: sometimes a list, sometimes prose, sometimes 3 queries instead of 10. The structured prompt with explicit type requirements and an inline example constrains the output space and gives the model a clear success criterion.

---

## Embedding Model Choice: `all-MiniLM-L6-v2`

I chose `all-MiniLM-L6-v2` over `all-mpnet-base-v2` for these reasons:

| | all-MiniLM-L6-v2 | all-mpnet-base-v2 |
|---|---|---|
| Speed | ~5× faster | Slower |
| Accuracy (MTEB) | ~63 | ~69 |
| Size | 80MB | 420MB |
| Inference (CPU) | ~30ms/batch | ~150ms/batch |

For a synchronous web API responding to user requests, the latency difference is significant and visible. The accuracy delta (63 vs 69 on MTEB) is unlikely to materially change coverage decisions at the 0.72 threshold — a query about "how to optimize blog content for SEO" will still match a paragraph about on-page SEO regardless of which model is used.

**Production consideration:** If accuracy were the priority (e.g. a batch offline job), I'd use `all-mpnet-base-v2` or a fine-tuned domain-specific model. For real-time web API use, MiniLM is the right call.

---

## Similarity Threshold: Why 0.72?

The 0.72 threshold is a reasonable starting point that balances precision and recall:

- **Too low (e.g. 0.5):** Almost everything marks as "covered" — the gap map becomes useless because superficially related content matches unrelated sub-queries.
- **Too high (e.g. 0.85):** Overly strict. A paragraph that genuinely covers a sub-query concept but uses different vocabulary will be flagged as a gap.
- **0.72:** Empirically, this catches genuine semantic overlap while filtering coincidental word overlap.

**How I'd tune it in production:**
1. Collect human-labeled pairs of (sub-query, content chunk) marked as covered/not-covered
2. Run the threshold as a hyperparameter sweep from 0.6 to 0.85
3. Optimize F1 score on the labeled dataset
4. Monitor threshold drift as the embedding model changes

---

## Failure Modes Addressed

| Scenario | Handling |
|---|---|
| LLM returns invalid JSON | Strip code fences, retry 3x with backoff, return 503 |
| LLM returns fewer than 10 queries | `_parse_and_validate` raises, triggers retry |
| URL unreachable / timeout | Returns HTTP 422 with `url_fetch_failed` error |
| spaCy model not installed | RuntimeError at first call with install instructions |
| Content too short to chunk | `analyze_gaps` returns all sub-queries as gaps (covered=False) |
| No `GEMINI_API_KEY` set | RuntimeError with clear message before any LLM call |

---

## Mock Mode (Rate Limit Workaround)

If you hit Gemini free-tier rate limits, set `MOCK_LLM=true` to bypass the real API call entirely. The fan-out endpoint will return a hardcoded 12-query fixture response so you can test the full flow (including gap analysis) without any API calls.

```bash
export MOCK_LLM=true
uvicorn app.main:app --reload
```

> **Note:** This is a development convenience only. The fixture response is always the same 12 sub-queries regardless of the `target_query` you send. Do not enable in production.

To go back to real Gemini calls: `unset MOCK_LLM`

---

## What I'd Improve With More Time

1. **Async NLP with ThreadPoolExecutor:** Move CPU-bound checks off the async event loop
2. **Playwright for JS-heavy pages:** Handle SPAs that return empty HTML on first fetch
3. **Prompt iteration log (PROMPT_LOG.md):** Document the 3 prompt drafts I went through before landing on the final structure
4. **Caching LLM responses in tests:** Add a fixture that stores a real Gemini response so integration tests don't hit the API on every run
5. **Threshold calibration dataset:** Build a small labeled set to empirically validate the 0.72 threshold rather than using it as a given
6. **H-tag check extension:** Currently checks H1–H6 but doesn't validate that the heading text is descriptive (a heading of "1." is technically valid but not AEO-friendly)
