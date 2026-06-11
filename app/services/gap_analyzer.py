from __future__ import annotations
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from app.models.schemas import SubQuery

SIMILARITY_THRESHOLD = 0.72
# all-MiniLM-L6-v2: ~5× faster than all-mpnet-base-v2 with minimal accuracy loss.
# For a synchronous web API where p99 latency matters, this is the right tradeoff.
MODEL_NAME = "all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _chunk_sentences(text: str, min_words: int = 5) -> list[str]:
    """Split text into sentence-level chunks, filtering trivially short ones."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.split()) >= min_words]


def analyze_gaps(sub_queries: list[SubQuery], content: str) -> list[SubQuery]:
    """
    For each sub-query, compute max cosine similarity against sentence chunks
    of content. Marks covered=True if similarity >= SIMILARITY_THRESHOLD.

    Uses normalize_embeddings=True so dot product == cosine similarity,
    avoiding the 'raw dot product on non-normalized vectors' red flag.
    """
    model = _get_model()
    chunks = _chunk_sentences(content)

    if not chunks:
        # No content to compare — all gaps
        return [
            SubQuery(type=sq.type, query=sq.query, covered=False, similarity_score=0.0)
            for sq in sub_queries
        ]

    # Encode all content chunks once (normalized)
    chunk_embeddings = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)

    # Encode all sub-queries in a single batch
    query_texts = [sq.query for sq in sub_queries]
    query_embeddings = model.encode(query_texts, normalize_embeddings=True, show_progress_bar=False)

    results: list[SubQuery] = []
    for sq, q_emb in zip(sub_queries, query_embeddings):
        # dot product on normalized vectors == cosine similarity
        similarities = np.dot(chunk_embeddings, q_emb)
        max_sim = float(np.max(similarities))
        results.append(
            SubQuery(
                type=sq.type,
                query=sq.query,
                covered=max_sim >= SIMILARITY_THRESHOLD,
                similarity_score=round(max_sim, 4),
            )
        )

    return results
