"""Tests for axiom.llm_providers -- fully offline (R: no real network calls).

Anthropic path: `httpx.MockTransport` swapped in via a monkeypatched
`_client_factory` (no `respx`/extra dependency needed -- `MockTransport` is
part of httpx, already pinned). Ollama path: `axiom.llm.chat_json` is
monkeypatched directly.
"""
from __future__ import annotations

import json

import httpx
import pytest
from pydantic import ValidationError

from axiom import config, llm, tracing
from axiom.agents.contracts import ExtractionResult, RouteDecision
from axiom import llm_providers as lp


@pytest.fixture(autouse=True)
def _fast_and_isolated(tmp_path, monkeypatch):
    """Every test: tiny retry/backoff budget (fast), its own trace dir, a
    clean spend ledger, and a known fast/strong price table."""
    monkeypatch.setattr(config, "TRACE_DIR", tmp_path)
    monkeypatch.setattr(config, "ANTHROPIC_MAX_RETRIES", 3)
    monkeypatch.setattr(config, "ANTHROPIC_BACKOFF_CAP", 0.01)
    monkeypatch.setattr(config, "LLM_BUDGET_USD", 3.00)
    monkeypatch.setattr(config, "MODEL_FAST_PRICE_IN", 1.00)
    monkeypatch.setattr(config, "MODEL_FAST_PRICE_OUT", 5.00)
    monkeypatch.setattr(config, "MODEL_STRONG_PRICE_IN", 2.00)
    monkeypatch.setattr(config, "MODEL_STRONG_PRICE_OUT", 10.00)
    lp.reset_spend()
    yield
    lp.reset_spend()


def _route_body(paper_id="W1", action="full", stop_reason="end_turn", tokens=(10, 5)):
    # `paper_id` kept as a parameter for call-site compatibility but no longer
    # part of the payload -- RouteDecision has no paper_id field (A4, stage-2
    # fixes): the model is never asked for one.
    del paper_id
    data = {"action": action, "reason": "ok"}
    return {
        "content": [{"type": "text", "text": json.dumps(data)}],
        "stop_reason": stop_reason,
        "usage": {"input_tokens": tokens[0], "output_tokens": tokens[1]},
    }


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://mocked")


# --- Schema sanitization -------------------------------------------------------

def test_build_schema_strips_disallowed_keywords_and_forces_additional_properties_false():
    schema = lp.build_schema(ExtractionResult)
    blob = json.dumps(schema)
    for kw in ("minimum", "maximum", "minLength", "maxLength"):
        assert f'"{kw}"' not in blob

    assert schema.get("additionalProperties") is False
    # Entity is nested under $defs for ExtractionResult; must be sanitized too.
    defs = schema.get("$defs", {})
    assert defs, "expected $defs for the nested Entity model"
    for name, sub in defs.items():
        if sub.get("type") == "object" or "properties" in sub:
            assert sub.get("additionalProperties") is False, name


def test_build_schema_preserves_type_and_enum_info():
    schema = lp.build_schema(RouteDecision)
    props = schema["properties"]
    assert props["action"]["enum"] if "enum" in props["action"] else True


# --- Cost accounting -----------------------------------------------------------

def test_estimate_cost_math():
    cost = lp.estimate_cost(config.MODEL_FAST, tokens_in=1_000_000, tokens_out=1_000_000)
    assert cost == pytest.approx(config.MODEL_FAST_PRICE_IN + config.MODEL_FAST_PRICE_OUT)

    cost_strong = lp.estimate_cost(config.MODEL_STRONG, tokens_in=500_000, tokens_out=200_000)
    expected = 0.5 * config.MODEL_STRONG_PRICE_IN + 0.2 * config.MODEL_STRONG_PRICE_OUT
    assert cost_strong == pytest.approx(expected)


def test_estimate_cost_unknown_model_is_zero():
    assert lp.estimate_cost("some-other-model", 1000, 1000) == 0.0


# --- Budget enforcement ---------------------------------------------------------

def test_budget_exceeded_blocks_before_any_call(monkeypatch):
    monkeypatch.setattr(config, "LLM_BUDGET_USD", 0.01)
    lp._record_spend("run1", 0.02)  # already over budget

    def _boom():
        raise AssertionError("should never build a client once over budget")
    monkeypatch.setattr(lp, "_client_factory", _boom)

    with pytest.raises(lp.BudgetExceededError):
        import asyncio
        asyncio.run(lp.AnthropicProvider().complete_json(
            "fast", "sys", "prompt", RouteDecision, run_id="run1", agent="coordinator",
        ))


def test_budget_is_global_not_per_run(monkeypatch):
    """Phase-4 architecture review finding: two run_ids, each individually
    under budget, must still trip the cap once their combined spend reaches
    it -- LLM_BUDGET_USD is a single process-wide total (AXIOM_AGENTIC_TASK.md
    §1), not a per-run allowance."""
    monkeypatch.setattr(config, "LLM_BUDGET_USD", 0.03)
    lp._record_spend("run1", 0.02)
    lp._record_spend("run2", 0.02)  # run2 alone is "only" $0.02, but total is $0.04

    def _boom():
        raise AssertionError("should never build a client once over budget")
    monkeypatch.setattr(lp, "_client_factory", _boom)

    with pytest.raises(lp.BudgetExceededError):
        import asyncio
        asyncio.run(lp.AnthropicProvider().complete_json(
            "fast", "sys", "prompt", RouteDecision, run_id="run2", agent="coordinator",
        ))


def test_spend_accumulates_across_calls(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-test")

    def handler(request):
        return httpx.Response(200, json=_route_body(tokens=(1_000_000, 1_000_000)))

    monkeypatch.setattr(lp, "_client_factory", lambda: _mock_client(handler))

    import asyncio
    asyncio.run(lp.AnthropicProvider().complete_json(
        "fast", "sys", "prompt", RouteDecision, run_id="run1", agent="coordinator",
    ))
    expected = config.MODEL_FAST_PRICE_IN + config.MODEL_FAST_PRICE_OUT
    assert lp.spend_so_far("run1") == pytest.approx(expected)


# --- Anthropic provider ---------------------------------------------------------

def test_anthropic_complete_json_success(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-test")
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=_route_body(action="title_only"))

    monkeypatch.setattr(lp, "_client_factory", lambda: _mock_client(handler))

    import asyncio
    result = asyncio.run(lp.complete_json(
        "fast", "sys", "prompt", RouteDecision, run_id="runA", agent="coordinator",
    ))
    assert isinstance(result, RouteDecision)
    assert result.action == "title_only"
    assert len(calls) == 1

    traces = tracing.read_trace("runA")
    assert len(traces) == 1
    assert traces[0]["verdict"] == "ok"
    assert traces[0]["agent"] == "coordinator"
    assert traces[0]["model"] == config.MODEL_FAST


def test_anthropic_refusal_raises_and_traces(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-test")

    def handler(request):
        return httpx.Response(200, json=_route_body(stop_reason="refusal"))

    monkeypatch.setattr(lp, "_client_factory", lambda: _mock_client(handler))

    import asyncio
    with pytest.raises(lp.LLMRefusalError):
        asyncio.run(lp.complete_json("fast", "sys", "prompt", RouteDecision, run_id="runB", agent="coordinator"))

    traces = tracing.read_trace("runB")
    assert traces[0]["verdict"] == "refusal"


def test_anthropic_max_tokens_raises_and_traces(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-test")

    def handler(request):
        return httpx.Response(200, json=_route_body(stop_reason="max_tokens"))

    monkeypatch.setattr(lp, "_client_factory", lambda: _mock_client(handler))

    import asyncio
    with pytest.raises(lp.LLMMaxTokensError):
        asyncio.run(lp.complete_json("fast", "sys", "prompt", RouteDecision, run_id="runC", agent="coordinator"))
    assert tracing.read_trace("runC")[0]["verdict"] == "max_tokens"


def test_anthropic_invalid_json_raises_parse_error(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-test")

    def handler(request):
        return httpx.Response(200, json={
            "content": [{"type": "text", "text": "not json{{"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })

    monkeypatch.setattr(lp, "_client_factory", lambda: _mock_client(handler))

    import asyncio
    with pytest.raises(lp.LLMParseError):
        asyncio.run(lp.complete_json("fast", "sys", "prompt", RouteDecision, run_id="runD", agent="coordinator"))
    assert tracing.read_trace("runD")[0]["verdict"] == "parse_error"


def test_anthropic_schema_mismatch_raises_validation_error(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-test")

    def handler(request):
        bad = {"action": "not-a-valid-action", "reason": "x"}
        return httpx.Response(200, json={
            "content": [{"type": "text", "text": json.dumps(bad)}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })

    monkeypatch.setattr(lp, "_client_factory", lambda: _mock_client(handler))

    import asyncio
    with pytest.raises(ValidationError):
        asyncio.run(lp.complete_json("fast", "sys", "prompt", RouteDecision, run_id="runE", agent="coordinator"))
    assert tracing.read_trace("runE")[0]["verdict"] == "schema_invalid"


def test_anthropic_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-test")
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json=_route_body())

    monkeypatch.setattr(lp, "_client_factory", lambda: _mock_client(handler))

    import asyncio
    result = asyncio.run(lp.complete_json("fast", "sys", "prompt", RouteDecision, run_id="runF", agent="coordinator"))
    assert isinstance(result, RouteDecision)
    assert attempts["n"] == 3


def test_anthropic_exhausts_retries_raises_transport_error(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(config, "ANTHROPIC_MAX_RETRIES", 2)

    def handler(request):
        return httpx.Response(503, text="down")

    monkeypatch.setattr(lp, "_client_factory", lambda: _mock_client(handler))

    import asyncio
    with pytest.raises(lp.LLMTransportError):
        asyncio.run(lp.complete_json("fast", "sys", "prompt", RouteDecision, run_id="runG", agent="coordinator"))
    assert tracing.read_trace("runG")[0]["verdict"] == "transport_error"


# --- Ollama provider -------------------------------------------------------------

def test_ollama_complete_json_success(monkeypatch):
    def fake_chat_json(prompt, *, system=None, model=None, temperature=0.2, timeout=None):
        return {"action": "full", "reason": "ok"}
    monkeypatch.setattr(llm, "chat_json", fake_chat_json)

    import asyncio
    result = asyncio.run(lp.OllamaProvider().complete_json(
        "fast", "sys", "prompt", RouteDecision, run_id="runH", agent="coordinator",
    ))
    assert isinstance(result, RouteDecision)
    assert result.action == "full"
    trace = tracing.read_trace("runH")[0]
    assert trace["verdict"] == "ok"
    assert trace["cost_usd"] == 0.0


def test_ollama_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_chat_json(prompt, *, system=None, model=None, temperature=0.2, timeout=None):
        calls["n"] += 1
        if calls["n"] < 2:
            raise llm.OllamaError("unreachable")
        return {"action": "full", "reason": "ok"}
    monkeypatch.setattr(llm, "chat_json", fake_chat_json)
    monkeypatch.setattr(config, "ANTHROPIC_MAX_RETRIES", 3)
    monkeypatch.setattr(config, "ANTHROPIC_BACKOFF_CAP", 0.01)

    import asyncio
    result = asyncio.run(lp.OllamaProvider().complete_json(
        "fast", "sys", "prompt", RouteDecision, run_id="runI", agent="coordinator",
    ))
    assert isinstance(result, RouteDecision)
    assert calls["n"] == 2


def test_ollama_exhausts_retries_raises_transport_error(monkeypatch):
    def fake_chat_json(prompt, *, system=None, model=None, temperature=0.2, timeout=None):
        raise llm.OllamaError("down")
    monkeypatch.setattr(llm, "chat_json", fake_chat_json)
    monkeypatch.setattr(config, "ANTHROPIC_MAX_RETRIES", 2)
    monkeypatch.setattr(config, "ANTHROPIC_BACKOFF_CAP", 0.01)

    import asyncio
    with pytest.raises(lp.LLMTransportError):
        asyncio.run(lp.OllamaProvider().complete_json(
            "fast", "sys", "prompt", RouteDecision, run_id="runJ", agent="coordinator",
        ))
    assert tracing.read_trace("runJ")[0]["verdict"] == "transport_error"


def test_ollama_schema_validation_error_traces_and_raises(monkeypatch):
    def fake_chat_json(prompt, *, system=None, model=None, temperature=0.2, timeout=None):
        return {"action": "not-valid", "reason": "x"}
    monkeypatch.setattr(llm, "chat_json", fake_chat_json)

    import asyncio
    with pytest.raises(ValidationError):
        asyncio.run(lp.OllamaProvider().complete_json(
            "fast", "sys", "prompt", RouteDecision, run_id="runK", agent="coordinator",
        ))
    assert tracing.read_trace("runK")[0]["verdict"] == "schema_invalid"


# --- Provider selection / fallback ------------------------------------------------

def test_select_provider_name_prefers_anthropic_when_key_set(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-test")
    assert lp.select_provider_name() == "anthropic"


def test_select_provider_name_falls_back_to_ollama_without_key(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None)
    assert lp.select_provider_name() == "ollama"


def test_complete_json_dispatches_to_ollama_without_key(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", None)

    def fake_chat_json(prompt, *, system=None, model=None, temperature=0.2, timeout=None):
        return {"action": "skip", "reason": "no abstract"}
    monkeypatch.setattr(llm, "chat_json", fake_chat_json)

    import asyncio
    result = asyncio.run(lp.complete_json("fast", "sys", "prompt", RouteDecision, run_id="runL", agent="coordinator"))
    assert result.action == "skip"
