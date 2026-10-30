"""Async LLM provider facade (R5/R6, docs/agentic/ARCHITECTURE.md §1).

`complete_json(tier, system, prompt, model_cls, ...)` is the one entry point
every agent uses. Two providers implement it:

- `anthropic` (default when `config.ANTHROPIC_API_KEY` is set): plain
  `httpx.AsyncClient` against the Messages API -- no `anthropic` SDK, per the
  repo's minimal-deps discipline (requirements.txt only pins httpx).
  `fast` -> `config.MODEL_FAST`, `strong` -> `config.MODEL_STRONG`.
  Structured output via `output_config.format = {"type": "json_schema",
  "schema": ...}`, built from `model_cls.model_json_schema()` with
  `additionalProperties: false` forced on every object and numeric/length
  constraints (`minimum`, `maximum`, `minLength`, `maxLength`, ...) stripped
  -- those stay enforced by Pydantic on the parsed response; the verifier
  (`axiom/agents/verifier.py`) is the backstop, not the API (OD23 #4).
- `ollama` (fallback, no key): wraps the existing `axiom/llm.py` client
  (OD14) with the same `complete_json` interface. Local, so $0 cost; token
  counts are a word-count estimate since `axiom/llm.py`'s wrapper doesn't
  surface Ollama's `eval_count`/`prompt_eval_count` fields.

Provider selection follows `config.LLM_PROVIDER_PRIORITY`: `anthropic` iff
`config.ANTHROPIC_API_KEY` is truthy, else `ollama` (mirrors OD14's no-key
philosophy -- the pipeline always runs).

Every call: timeout (`ANTHROPIC_TIMEOUT`), retry with exponential backoff +
jitter on 429/5xx up to `ANTHROPIC_MAX_RETRIES` (capped at
`ANTHROPIC_BACKOFF_CAP`), token+cost accounting against `MODEL_FAST_PRICE_*`
/`MODEL_STRONG_PRICE_*`, a hard stop once `config.LLM_BUDGET_USD` is spent for
a run, and exactly one `axiom/tracing.py` line per call attempt -- success or
failure.
"""
from __future__ import annotations

import asyncio
import json
import random
import threading
import time
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from axiom import config, llm, tracing

ModelT = TypeVar("ModelT", bound=BaseModel)

_ANTHROPIC_VERSION = "2023-06-01"
_DEFAULT_MAX_TOKENS = 4096

# Keywords the Anthropic json_schema format rejects. Pydantic keeps enforcing
# them on the parsed response (that's the verifier's "schema" check).
_STRIP_KEYWORDS = (
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "minLength", "maxLength", "minItems", "maxItems", "multipleOf", "pattern",
)


# --- Errors ------------------------------------------------------------------

class LLMError(RuntimeError):
    """Base for every provider-layer failure."""


class LLMRefusalError(LLMError):
    """The model refused (stop_reason == 'refusal')."""


class LLMMaxTokensError(LLMError):
    """The model hit the output token cap before finishing (stop_reason == 'max_tokens')."""


class LLMParseError(LLMError):
    """The response text was not valid JSON."""


class LLMTransportError(LLMError):
    """Network error or 429/5xx that survived every retry."""


class BudgetExceededError(LLMError):
    """`config.LLM_BUDGET_USD` has already been spent for this run."""


# --- Schema sanitization (Anthropic json_schema format) ----------------------

def _sanitize_schema(node: Any) -> Any:
    """Recursively force `additionalProperties: false` on every object and
    strip keywords the Anthropic structured-output API rejects. Mutates and
    returns `node` (also walks `$defs` for nested Pydantic models)."""
    if isinstance(node, dict):
        for kw in _STRIP_KEYWORDS:
            node.pop(kw, None)
        if node.get("type") == "object" or "properties" in node:
            node["additionalProperties"] = False
        for value in node.values():
            _sanitize_schema(value)
    elif isinstance(node, list):
        for item in node:
            _sanitize_schema(item)
    return node


def build_schema(model_cls: type[BaseModel]) -> dict:
    """`model_cls.model_json_schema()`, sanitized for the Anthropic API."""
    schema = model_cls.model_json_schema()
    return _sanitize_schema(schema)


# --- Cost accounting -----------------------------------------------------------

def _price_for(model: str) -> tuple[float, float]:
    if model == config.MODEL_FAST:
        return config.MODEL_FAST_PRICE_IN, config.MODEL_FAST_PRICE_OUT
    if model == config.MODEL_STRONG:
        return config.MODEL_STRONG_PRICE_IN, config.MODEL_STRONG_PRICE_OUT
    return 0.0, 0.0


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    price_in, price_out = _price_for(model)
    return (tokens_in / 1_000_000) * price_in + (tokens_out / 1_000_000) * price_out


_spend_lock = threading.Lock()
_spend_by_run: dict[str, float] = {}


def spend_so_far(run_id: str) -> float:
    with _spend_lock:
        return _spend_by_run.get(run_id, 0.0)


def reset_spend(run_id: str | None = None) -> None:
    """Test/bench helper: clear tracked spend for one run, or all runs."""
    with _spend_lock:
        if run_id is None:
            _spend_by_run.clear()
        else:
            _spend_by_run.pop(run_id, None)


def _ensure_budget(run_id: str) -> None:
    with _spend_lock:
        total = _spend_by_run.get(run_id, 0.0)
    if total >= config.LLM_BUDGET_USD:
        raise BudgetExceededError(
            f"run {run_id!r} already spent ${total:.4f} >= "
            f"LLM_BUDGET_USD ${config.LLM_BUDGET_USD:.2f}"
        )


def _record_spend(run_id: str, cost_usd: float) -> None:
    with _spend_lock:
        _spend_by_run[run_id] = _spend_by_run.get(run_id, 0.0) + cost_usd


# --- Backoff -------------------------------------------------------------------

def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff with jitter, capped at ANTHROPIC_BACKOFF_CAP.

    `attempt` is 1-based (first retry -> attempt=1).
    """
    base = min(config.ANTHROPIC_BACKOFF_CAP, 2 ** (attempt - 1))
    return base * (0.5 + random.random())


# --- Anthropic provider --------------------------------------------------------

def _client_factory() -> httpx.AsyncClient:
    """Build the httpx client used for Anthropic calls. A separate factory
    function (rather than inlining `httpx.AsyncClient(...)`) so tests can
    monkeypatch it to inject a `MockTransport`."""
    return httpx.AsyncClient(base_url=config.ANTHROPIC_BASE_URL, timeout=config.ANTHROPIC_TIMEOUT)


async def _anthropic_post(client: httpx.AsyncClient, payload: dict) -> httpx.Response:
    """POST /v1/messages with retry + backoff on 429/5xx or network errors."""
    headers = {
        "x-api-key": config.ANTHROPIC_API_KEY or "",
        "anthropic-version": _ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    last_exc: Exception | None = None
    max_attempts = max(1, config.ANTHROPIC_MAX_RETRIES)
    for attempt in range(1, max_attempts + 1):
        try:
            resp = await client.post("/v1/messages", json=payload, headers=headers)
        except httpx.HTTPError as exc:
            last_exc = exc
        else:
            if resp.status_code == 429 or resp.status_code >= 500:
                last_exc = LLMTransportError(
                    f"HTTP {resp.status_code} from Anthropic: {resp.text[:300]!r}"
                )
            else:
                return resp
        if attempt < max_attempts:
            await asyncio.sleep(_backoff_seconds(attempt))
    raise LLMTransportError(
        f"Anthropic request failed after {max_attempts} attempts: {last_exc}"
    ) from last_exc


class AnthropicProvider:
    name = "anthropic"
    _tier_models = {"fast": config.MODEL_FAST, "strong": config.MODEL_STRONG}

    async def complete_json(
        self,
        tier: str,
        system: str,
        prompt: str,
        model_cls: type[ModelT],
        *,
        run_id: str,
        agent: str,
        attempt: int = 1,
        temperature: float | None = None,
    ) -> ModelT:
        model = self._tier_models[tier]
        _ensure_budget(run_id)

        schema = build_schema(model_cls)
        payload = {
            "model": model,
            "max_tokens": _DEFAULT_MAX_TOKENS,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        }
        if temperature is not None:
            payload["temperature"] = temperature

        start = time.perf_counter()
        client = _client_factory()
        try:
            resp = await _anthropic_post(client, payload)
        except LLMTransportError as exc:
            tracing.log_call(
                run_id, agent=agent, model=model, verdict="transport_error",
                attempt=attempt, latency_s=time.perf_counter() - start,
                reasons=[str(exc)],
            )
            raise
        finally:
            await client.aclose()
        latency_s = time.perf_counter() - start

        body = resp.json()
        usage = body.get("usage", {})
        tokens_in = int(usage.get("input_tokens", 0))
        tokens_out = int(usage.get("output_tokens", 0))
        cost_usd = estimate_cost(model, tokens_in, tokens_out)

        stop_reason = body.get("stop_reason")
        if stop_reason == "refusal":
            tracing.log_call(
                run_id, agent=agent, model=model, verdict="refusal", attempt=attempt,
                tokens_in=tokens_in, tokens_out=tokens_out, latency_s=latency_s,
                cost_usd=cost_usd, reasons=["model refused to answer"],
            )
            _record_spend(run_id, cost_usd)
            raise LLMRefusalError(f"{agent}: model refused (run {run_id})")
        if stop_reason == "max_tokens":
            tracing.log_call(
                run_id, agent=agent, model=model, verdict="max_tokens", attempt=attempt,
                tokens_in=tokens_in, tokens_out=tokens_out, latency_s=latency_s,
                cost_usd=cost_usd, reasons=["hit max_tokens before finishing"],
            )
            _record_spend(run_id, cost_usd)
            raise LLMMaxTokensError(f"{agent}: hit max_tokens (run {run_id})")

        text = _extract_text(body)
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            tracing.log_call(
                run_id, agent=agent, model=model, verdict="parse_error", attempt=attempt,
                tokens_in=tokens_in, tokens_out=tokens_out, latency_s=latency_s,
                cost_usd=cost_usd, reasons=[str(exc)],
            )
            _record_spend(run_id, cost_usd)
            raise LLMParseError(f"{agent}: invalid JSON from model: {exc}") from exc

        try:
            result = model_cls(**data)
        except ValidationError as exc:
            tracing.log_call(
                run_id, agent=agent, model=model, verdict="schema_invalid", attempt=attempt,
                tokens_in=tokens_in, tokens_out=tokens_out, latency_s=latency_s,
                cost_usd=cost_usd, reasons=[str(e) for e in exc.errors()],
            )
            _record_spend(run_id, cost_usd)
            raise

        tracing.log_call(
            run_id, agent=agent, model=model, verdict="ok", attempt=attempt,
            tokens_in=tokens_in, tokens_out=tokens_out, latency_s=latency_s,
            cost_usd=cost_usd,
        )
        _record_spend(run_id, cost_usd)
        return result


def _extract_text(body: dict) -> str:
    for block in body.get("content", []):
        if block.get("type") == "text":
            return block.get("text", "")
    return ""


# --- Ollama provider (fallback, OD14) ------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Crude word-count proxy. Ollama is $0 regardless; this is only to keep
    the trace line's tokens_in/out fields populated for p50/p95-style
    inspection, not for billing."""
    return len(text.split())


class OllamaProvider:
    name = "ollama"
    _tier_models = {"fast": config.OLLAMA_MODEL, "strong": config.OLLAMA_MODEL}

    async def complete_json(
        self,
        tier: str,
        system: str,
        prompt: str,
        model_cls: type[ModelT],
        *,
        run_id: str,
        agent: str,
        attempt: int = 1,
        temperature: float | None = None,
    ) -> ModelT:
        model = self._tier_models[tier]
        _ensure_budget(run_id)  # local model is free, but a shared budget guard costs nothing

        start = time.perf_counter()
        last_exc: Exception | None = None
        max_attempts = max(1, config.ANTHROPIC_MAX_RETRIES)
        data: dict | None = None
        kwargs = {"system": system, "model": model}
        if temperature is not None:
            kwargs["temperature"] = temperature
        for try_num in range(1, max_attempts + 1):
            try:
                data = await asyncio.to_thread(llm.chat_json, prompt, **kwargs)
                last_exc = None
                break
            except llm.OllamaError as exc:
                last_exc = exc
                if try_num < max_attempts:
                    await asyncio.sleep(_backoff_seconds(try_num))
        latency_s = time.perf_counter() - start

        tokens_in = _estimate_tokens(system) + _estimate_tokens(prompt)

        if data is None:
            tracing.log_call(
                run_id, agent=agent, model=model, verdict="transport_error",
                attempt=attempt, tokens_in=tokens_in, latency_s=latency_s,
                reasons=[str(last_exc)],
            )
            raise LLMTransportError(f"{agent}: Ollama unreachable after retries") from last_exc

        tokens_out = _estimate_tokens(json.dumps(data))
        try:
            result = model_cls(**data)
        except ValidationError as exc:
            tracing.log_call(
                run_id, agent=agent, model=model, verdict="schema_invalid", attempt=attempt,
                tokens_in=tokens_in, tokens_out=tokens_out, latency_s=latency_s,
                reasons=[str(e) for e in exc.errors()],
            )
            raise

        tracing.log_call(
            run_id, agent=agent, model=model, verdict="ok", attempt=attempt,
            tokens_in=tokens_in, tokens_out=tokens_out, latency_s=latency_s, cost_usd=0.0,
        )
        return result


# --- Dispatch -------------------------------------------------------------------

def select_provider_name() -> str:
    """Which provider `complete_json` will use right now, per
    `config.LLM_PROVIDER_PRIORITY` (anthropic iff an API key is set)."""
    for name in config.LLM_PROVIDER_PRIORITY:
        if name == "anthropic" and config.ANTHROPIC_API_KEY:
            return "anthropic"
        if name == "ollama":
            return "ollama"
    return "ollama"


def _get_provider():
    return AnthropicProvider() if select_provider_name() == "anthropic" else OllamaProvider()


async def complete_json(
    tier: str,
    system: str,
    prompt: str,
    model_cls: type[ModelT],
    *,
    run_id: str = "adhoc",
    agent: str = "llm",
    attempt: int = 1,
    temperature: float | None = None,
) -> ModelT:
    """Provider-agnostic structured completion. Raises `LLMError` subclasses
    on transport/refusal/parse failure, `pydantic.ValidationError` when the
    model's JSON doesn't satisfy `model_cls` (the caller's retry loop builds
    `VerifierError`s from `exc.errors()`, per R4). `temperature` is optional
    (coordinator/extraction leave it at the provider's default; synthesis's
    hypothesis generator passes a low value per OD16's generator-temperature
    rule)."""
    provider = _get_provider()
    return await provider.complete_json(
        tier, system, prompt, model_cls, run_id=run_id, agent=agent,
        attempt=attempt, temperature=temperature,
    )
