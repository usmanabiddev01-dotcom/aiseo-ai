"""
Tests for Fan-out engine JSON parsing and validation logic.
The LLM call is mocked — these tests verify the parsing/retry/validation
logic in isolation from the actual Gemini API.
"""
import json
import pytest
from unittest.mock import MagicMock, patch
from app.services.fanout_engine import _parse_and_validate, _strip_code_fences, generate_sub_queries


# ── _strip_code_fences ────────────────────────────────────────────────────────

def test_strip_code_fences_removes_json_block():
    raw = "```json\n{\"sub_queries\": []}\n```"
    assert _strip_code_fences(raw) == '{"sub_queries": []}'

def test_strip_code_fences_removes_plain_block():
    raw = "```\n{\"sub_queries\": []}\n```"
    assert _strip_code_fences(raw) == '{"sub_queries": []}'

def test_strip_code_fences_no_op_on_clean_json():
    raw = '{"sub_queries": []}'
    assert _strip_code_fences(raw) == raw


# ── _parse_and_validate ───────────────────────────────────────────────────────

def _make_valid_response(n_per_type: int = 2) -> str:
    types = ["comparative", "feature_specific", "use_case", "trust_signals", "how_to", "definitional"]
    items = []
    for t in types:
        for i in range(n_per_type):
            items.append({"type": t, "query": f"Example {t} query {i+1}"})
    return json.dumps({"sub_queries": items})


def test_parse_valid_response_returns_sub_queries():
    raw = _make_valid_response(2)
    result = _parse_and_validate(raw)
    assert len(result) == 12
    assert all(sq.type in {"comparative", "feature_specific", "use_case", "trust_signals", "how_to", "definitional"} for sq in result)


def test_parse_caps_at_15_sub_queries():
    types = ["comparative", "feature_specific", "use_case", "trust_signals", "how_to", "definitional"]
    items = [{"type": t, "query": f"Query {i}"} for i, t in enumerate(types * 5)]  # 30 items
    raw = json.dumps({"sub_queries": items})
    result = _parse_and_validate(raw)
    assert len(result) <= 15


def test_parse_raises_on_missing_key():
    raw = json.dumps({"wrong_key": []})
    with pytest.raises(ValueError, match="missing 'sub_queries'"):
        _parse_and_validate(raw)


def test_parse_raises_on_too_few_sub_queries():
    items = [{"type": "comparative", "query": "only one"}]
    raw = json.dumps({"sub_queries": items})
    with pytest.raises(ValueError, match="Expected"):
        _parse_and_validate(raw)


def test_parse_raises_on_unknown_type():
    types = ["comparative", "feature_specific", "use_case", "trust_signals", "how_to", "definitional"]
    items = [{"type": t, "query": f"q{i}"} for i, t in enumerate(types * 2)]
    items.append({"type": "hallucinated_type", "query": "bad"})
    raw = json.dumps({"sub_queries": items})
    with pytest.raises(ValueError, match="Unknown type"):
        _parse_and_validate(raw)


def test_parse_raises_on_missing_types():
    """If a type is entirely absent from the response, should raise."""
    items = [{"type": "comparative", "query": f"q{i}"} for i in range(12)]
    raw = json.dumps({"sub_queries": items})
    with pytest.raises(ValueError, match="Missing sub-query types"):
        _parse_and_validate(raw)


def test_parse_raises_on_invalid_json():
    with pytest.raises(json.JSONDecodeError):
        _parse_and_validate("not json at all {{{")


# ── generate_sub_queries with mocked LLM ─────────────────────────────────────

@patch("app.services.fanout_engine.genai")
def test_generate_sub_queries_success(mock_genai):
    """Valid LLM response → returns list of SubQuery objects."""
    valid_json = _make_valid_response(2)
    mock_response = MagicMock()
    mock_response.text = valid_json
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    mock_genai.Client.return_value = mock_client

    import os
    os.environ["GEMINI_API_KEY"] = "test-key"

    result = generate_sub_queries("best AI writing tool for SEO")
    assert len(result) == 12
    assert result[0].query is not None


@patch("app.services.fanout_engine.genai")
@patch("app.services.fanout_engine.time.sleep", return_value=None)
def test_generate_sub_queries_retries_on_bad_json(mock_sleep, mock_genai):
    """Malformed JSON response → retries 3 times → raises RuntimeError."""
    mock_response = MagicMock()
    mock_response.text = "not valid json {{{"
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    mock_genai.Client.return_value = mock_client

    import os
    os.environ["GEMINI_API_KEY"] = "test-key"
    os.environ.pop("MOCK_LLM", None)

    with pytest.raises(RuntimeError, match="JSONDecodeError on attempt 3"):
        generate_sub_queries("best AI writing tool for SEO")

    assert mock_client.models.generate_content.call_count == 3


@patch("app.services.fanout_engine.genai")
@patch("app.services.fanout_engine.time.sleep", return_value=None)
def test_generate_sub_queries_succeeds_on_second_attempt(mock_sleep, mock_genai):
    """First attempt fails, second succeeds → returns valid result."""
    valid_json = _make_valid_response(2)

    bad_response = MagicMock()
    bad_response.text = "bad json"

    good_response = MagicMock()
    good_response.text = valid_json

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [bad_response, good_response]
    mock_genai.Client.return_value = mock_client

    import os
    os.environ["GEMINI_API_KEY"] = "test-key"
    os.environ.pop("MOCK_LLM", None)

    result = generate_sub_queries("best AI writing tool for SEO")
    assert len(result) == 12
    assert mock_client.models.generate_content.call_count == 2
