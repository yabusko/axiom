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
  `max_tokens` is looked up per call from `config.AGENT_MAX_TOKENS` by the
  caller-supplied `agent_type` (falls back to `config._DEFAULT_MAX_TOKENS`
  when omitted or unrecognized) -- routing/extraction/synthesis calls don't
  all need the same output-length cap (C4).
- `ollama` (fallback, no key): wraps the existing `axiom/llm.py` client
  (OD14) with the same `complete_json` interface. Local, so $0 cost; token
  counts are a word-count estimate since `axiom/llm.py`'s wrapper doesn't
  surface Ollama's `eval_count`/`prompt_eval_count` fields.

Provider selection follows `config.LLM_PROVIDER_PRIORITY`: `anthropic` iff
`config.ANTHROPIC_API_KEY` is truthy, else `ollama` (mirrors OD14's no-key
philosophy -- the pipeline always runs).

Every call: timeout (`ANTHROPIC_TIMEOUT`), retry with exponential backoff +
jitter (or the server's `retry-after`, when given) on 408/409/429/5xx up to
`ANTHROPIC_MAX_RETRIES` (capped at `ANTHROPIC_BACKOFF_CAP`) -- any other 4xx
raises `LLMHTTPError` immediately, no retry; token+cost accounting against
`MODEL_FAST_PRICE_*`/`MODEL_STRONG_PRICE_*`; a hard stop once
`config.LLM_BUDGET_USD` would be exceeded (checked against a durable,
cross-run total plus an in-process reservation ledger when a caller passes
a `conn`, else an in-process total only -- see `_reserve_budget`); and
exactly one `axiom/tracing.py` line per call attempt -- success or failure.

Two optional parameters support run-scoped callers (`axiom/agents/pipeline.py`):
`client` (an `httpx.AsyncClient` to reuse across every call in a run instead
of opening/closing one per call) and `conn` (a sqlite3 connection so the
budget check reads `db.total_llm_spend` -- the durable, restart-proof figure
-- instead of only this process's in-memory total). Both default to the
per-call/in-process behavior when omitted, so existing callers/tests are
unaffected.
"""
from __future__ import annotations

import asyncio
import json
import random
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from axiom import config, db, llm, tracing

ModelT = TypeVar("ModelT", bound=BaseModel)

_ANTHROPIC_VERSION = "2023-06-01"

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
    """Network error or 408/409/429/5xx that survived every retry."""


class LLMHTTPError(LLMError):
    """A non-retryable HTTP error from the Anthropic API (C1).

    Distinct from `LLMParseError`: the response body here is the API's error
    envelope (`{"type": "error", "error": {"type": ..., "message": ...}}`),
    never the model's JSON output, so it must never be run through
    `json.loads` as if it were one -- a 4xx like 400/401/403/404/422 means
    the request itself was rejected (bad key, malformed payload, ...), not
    that the model produced bad output. `status`/`type`/`message` are parsed
    straight out of that envelope (falling back to the raw response text if
    the body isn't that shape).
    """

    def __init__(self, status: int, error_type: str, message: str):
        self.status = status
        self.type = error_type
        self.message = message
        super().__init__(f"HTTP {status} ({error_type}): {message}")


class BudgetExceededError(LLMError):
    """Dispatching this call would push committed + reserved spend past
    `config.LLM_BUDGET_USD` (see `_reserve_budget`)."""


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


def _max_tokens_for(agent_type: str | None) -> int:
    """C4: per-agent-type output cap, `config.AGENT_MAX_TOKENS[agent_type]`,
    falling back to `config._DEFAULT_MAX_TOKENS` when `agent_type` is `None`
    or isn't a recognized key -- routing needs far fewer tokens than a
    synthesis pitch, and a flat 4096 everywhere both overpays on the budget
    estimate and under-caps nothing usefully."""
    return config.AGENT_MAX_TOKENS.get(agent_type, config._DEFAULT_MAX_TOKENS)


# --- Cost accounting -----------------------------------------------------------

def _price_for(model: str) -> tuple[float, float]:
    if model == config.MODEL_FAST:
        return config.MODEL_FAST_PRICE_IN, config.MODEL_FAST_PRICE_OUT
    if model == config.MODEL_STRONG:
        return config.MODEL_STRONG_PRICE_IN, config.MODEL_STRONG_PRICE_OUT
    if model == config.MODEL_OPUS:
        return config.MODEL_OPUS_PRICE_IN, config.MODEL_OPUS_PRICE_OUT
    return 0.0, 0.0


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    price_in, price_out = _price_for(model)
    return (tokens_in / 1_000_000) * price_in + (tokens_out / 1_000_000) * price_out


def _estimate_worst_case_cost(model: str, prompt_chars: int, max_tokens: int) -> float:
    """Pre-call estimate used only to reserve budget (C3): input tokens via a
    chars/4 proxy (no new tokenizer dependency), output tokens assumed to hit
    the call's `max_tokens` cap (the true worst case -- the API will never
    return more than that). Always >= the eventual real cost for a
    successful call, so reserving this amount can only be conservative, never
    let a call through that the real spend would have blocked."""
    price_in, price_out = _price_for(model)
    tokens_in_est = max(0, round(prompt_chars / 4))
    return (tokens_in_est / 1_000_000) * price_in + (max_tokens / 1_000_000) * price_out


_spend_lock = threading.Lock()
_spend_by_run: dict[str, float] = {}
_spend_total: float = 0.0  # process-wide, across every run_id -- the in-process
                           # fallback total when a caller doesn't pass a `conn`

_budget_lock = asyncio.Lock()
_reserved_total: float = 0.0  # in-process, currently-outstanding reservations (C3)


def spend_so_far(run_id: str) -> float:
    with _spend_lock:
        return _spend_by_run.get(run_id, 0.0)


def total_spend() -> float:
    """Process-wide spend across every run, tracked in-process only. Used as
    the committed-spend figure when a caller doesn't pass a `conn` into
    `complete_json` (see `_committed_spend`); when a `conn` IS passed,
    `db.total_llm_spend(conn)` -- durable across restarts -- is what
    `LLM_BUDGET_USD` actually caps (C3)."""
    with _spend_lock:
        return _spend_total


def reset_spend(run_id: str | None = None) -> None:
    """Test/bench helper: clear tracked spend for one run (and its share of
    the total), or everything -- also clears the in-process reservation
    ledger, since a leftover reservation from a previous test would corrupt
    the next test's budget check."""
    global _spend_total, _reserved_total
    with _spend_lock:
        if run_id is None:
            _spend_by_run.clear()
            _spend_total = 0.0
        else:
            _spend_total -= _spend_by_run.pop(run_id, 0.0)
    _reserved_total = 0.0


def _record_spend(run_id: str, cost_usd: float) -> None:
    global _spend_total
    with _spend_lock:
        _spend_by_run[run_id] = _spend_by_run.get(run_id, 0.0) + cost_usd
        _spend_total += cost_usd


def _committed_spend(conn: sqlite3.Connection | None) -> float:
    """The durable, cross-run committed-spend figure `LLM_BUDGET_USD` caps
    (C3) when `conn` is given; otherwise falls back to this process's
    in-memory total (today's behavior, for callers/tests that don't wire a
    connection through `complete_json`)."""
    if conn is not None:
        return db.total_llm_spend(conn)
    return total_spend()


async def _reserve_budget(conn: sqlite3.Connection | None, run_id: str, estimated_cost: float) -> None:
    """Check-and-reserve, serialized through `_budget_lock` (C3): concurrent
    calls under `config.AGENT_CONCURRENCY` all go through this one at a time,
    so none of them can read a stale "under budget" snapshot and all pass the
    check before any of them settles. `_reserved_total` tracks cost that's
    been admitted but not yet settled (call still in flight); release it via
    `_release_budget` once the call finishes, success or failure."""
    global _reserved_total
    async with _budget_lock:
        committed = _committed_spend(conn)
        if committed + _reserved_total + estimated_cost > config.LLM_BUDGET_USD:
            raise BudgetExceededError(
                f"run {run_id!r}: committed ${committed:.4f} + in-flight reservations "
                f"${_reserved_total:.4f} + this call's worst-case ${estimated_cost:.4f} "
                f"would exceed LLM_BUDGET_USD ${config.LLM_BUDGET_USD:.2f}"
            )
        _reserved_total += estimated_cost


async def _release_budget(estimated_cost: float) -> None:
    global _reserved_total
    async with _budget_lock:
        _reserved_total = max(0.0, _reserved_total - estimated_cost)


def _settle_spend(
    conn: sqlite3.Connection | None, run_id: str, *, model: str, tokens_in: int,
    tokens_out: int, cost_usd: float,
) -> None:
    """Record a call's REAL cost once it has settled (success or failure --
    a failed call that still consumed tokens, e.g. a refusal or a schema
    mismatch, still gets billed). Always updates the in-process total
    (`spend_so_far`/`total_spend` keep working for callers that don't pass a
    `conn`); additionally writes a durable `llm_spend` row when `conn` is
    given, which is what makes the budget check restart-proof (C3)."""
    _record_spend(run_id, cost_usd)
    if conn is not None:
        db.record_llm_spend(
            conn, run_id, ts=datetime.now(timezone.utc).isoformat(),
            model=model, tokens_in=tokens_in, tokens_out=tokens_out, cost=cost_usd,
        )


# --- Backoff -------------------------------------------------------------------

def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff with jitter, capped at ANTHROPIC_BACKOFF_CAP.

    `attempt` is 1-based (first retry -> attempt=1).
    """
    base = min(config.ANTHROPIC_BACKOFF_CAP, 2 ** (attempt - 1))
    return base * (0.5 + random.random())


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a `retry-after` header value (seconds, the form Anthropic's
    429/5xx/529 responses use) into a wait time; `None` if missing or not a
    plain number (an HTTP-date form falls back to blind backoff rather than
    crashing on a header this never actually sends)."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


# --- Anthropic provider --------------------------------------------------------

# C1: retryable transport-layer statuses -- 429 (rate limit), 408/409 (request
# timeout / conflict, both safe to resend unchanged), and anything >=500
# (server error, including Anthropic's 529 "overloaded"). Every OTHER 4xx
# (400/401/403/404/422/...) means the request itself was rejected and will
# never succeed by resending it -- those raise `LLMHTTPError` immediately.
_RETRYABLE_STATUSES = frozenset({408, 409, 429})


def _client_factory() -> httpx.AsyncClient:
    """Build the httpx client used for Anthropic calls. A separate factory
    function (rather than inlining `httpx.AsyncClient(...)`) so tests can
    monkeypatch it to inject a `MockTransport`."""
    return httpx.AsyncClient(base_url=config.ANTHROPIC_BASE_URL, timeout=config.ANTHROPIC_TIMEOUT)


def _build_http_error(resp: httpx.Response) -> LLMHTTPError:
    """Parse Anthropic's `{"type": "error", "error": {"type": ..., "message":
    ...}}` error-response shape into an `LLMHTTPError`; degrades to the raw
    response text if the body isn't that shape (never raises on a malformed
    error body)."""
    try:
        body = resp.json()
    except ValueError:
        body = {}
    err = body.get("error") if isinstance(body, dict) else None
    err = err if isinstance(err, dict) else {}
    error_type = err.get("type") or "unknown_error"
    message = err.get("message") or resp.text[:300]
    return LLMHTTPError(resp.status_code, error_type, message)


async def _anthropic_post(client: httpx.AsyncClient, payload: dict) -> httpx.Response:
    """POST /v1/messages with retry + backoff on 408/409/429/5xx or network
    errors (C1). Any other 4xx raises `LLMHTTPError` immediately -- no
    retry, and the body is never treated as model output. When a retried
    response carries a `retry-after` header, that value is used as the wait
    for the next attempt instead of blind exponential backoff."""
    headers = {
        "x-api-key": config.ANTHROPIC_API_KEY or "",
        "anthropic-version": _ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    last_exc: Exception | None = None
    max_attempts = max(1, config.ANTHROPIC_MAX_RETRIES)
    for attempt in range(1, max_attempts + 1):
        retry_after: float | None = None
        try:
            resp = await client.post("/v1/messages", json=payload, headers=headers)
        except httpx.HTTPError as exc:
            last_exc = exc
        else:
            if resp.status_code in _RETRYABLE_STATUSES or resp.status_code >= 500:
                last_exc = LLMTransportError(
                    f"HTTP {resp.status_code} from Anthropic: {resp.text[:300]!r}"
                )
                retry_after = _parse_retry_after(resp.headers.get("retry-after"))
            elif resp.status_code >= 400:
                raise _build_http_error(resp)
            else:
                return resp
        if attempt < max_attempts:
            wait = retry_after if retry_after is not None else _backoff_seconds(attempt)
            await asyncio.sleep(wait)
    raise LLMTransportError(
        f"Anthropic request failed after {max_attempts} attempts: {last_exc}"
    ) from last_exc


class AnthropicProvider:
    name = "anthropic"
    _tier_models = {
        "fast": config.MODEL_FAST, "strong": config.MODEL_STRONG, "opus": config.MODEL_OPUS,
    }

    async def complete_json(
        self,
        tier: str,
        system: str,
        prompt: str,
        model_cls: type[ModelT],
        *,
        run_id: str,
        agent: str,
        agent_type: str | None = None,
        paper_id: str | None = None,
        attempt: int = 1,
        temperature: float | None = None,
        client: httpx.AsyncClient | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> ModelT:
        model = self._tier_models[tier]
        max_tokens = _max_tokens_for(agent_type)
        estimated_cost = _estimate_worst_case_cost(model, len(system) + len(prompt), max_tokens)
        await _reserve_budget(conn, run_id, estimated_cost)

        schema = build_schema(model_cls)
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        }
        if temperature is not None:
            payload["temperature"] = temperature

        try:
            start = time.perf_counter()
            owns_client = client is None
            active_client = _client_factory() if owns_client else client
            try:
                try:
                    resp = await _anthropic_post(active_client, payload)
                except LLMTransportError as exc:
                    tracing.log_call(
                        run_id, agent=agent, model=model, paper_id=paper_id, verdict="transport_error",
                        attempt=attempt, latency_s=time.perf_counter() - start,
                        reasons=[str(exc)],
                    )
                    raise
                except LLMHTTPError as exc:
                    tracing.log_call(
                        run_id, agent=agent, model=model, paper_id=paper_id, verdict="http_error",
                        attempt=attempt, latency_s=time.perf_counter() - start,
                        reasons=[str(exc)],
                    )
                    raise
            finally:
                if owns_client:
                    await active_client.aclose()
            latency_s = time.perf_counter() - start

            body = resp.json()
            usage = body.get("usage", {})
            tokens_in = int(usage.get("input_tokens", 0))
            tokens_out = int(usage.get("output_tokens", 0))
            cost_usd = estimate_cost(model, tokens_in, tokens_out)

            stop_reason = body.get("stop_reason")
            if stop_reason == "refusal":
                tracing.log_call(
                    run_id, agent=agent, model=model, paper_id=paper_id, verdict="refusal", attempt=attempt,
                    tokens_in=tokens_in, tokens_out=tokens_out, latency_s=latency_s,
                    cost_usd=cost_usd, reasons=["model refused to answer"],
                )
                _settle_spend(conn, run_id, model=model, tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost_usd)
                raise LLMRefusalError(f"{agent}: model refused (run {run_id})")
            if stop_reason == "max_tokens":
                tracing.log_call(
                    run_id, agent=agent, model=model, paper_id=paper_id, verdict="max_tokens", attempt=attempt,
                    tokens_in=tokens_in, tokens_out=tokens_out, latency_s=latency_s,
                    cost_usd=cost_usd, reasons=["hit max_tokens before finishing"],
                )
                _settle_spend(conn, run_id, model=model, tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost_usd)
                raise LLMMaxTokensError(f"{agent}: hit max_tokens (run {run_id})")

            text = _extract_text(body)
            try:
                data = json.loads(text)
            except (json.JSONDecodeError, TypeError) as exc:
                tracing.log_call(
                    run_id, agent=agent, model=model, paper_id=paper_id, verdict="parse_error", attempt=attempt,
                    tokens_in=tokens_in, tokens_out=tokens_out, latency_s=latency_s,
                    cost_usd=cost_usd, reasons=[str(exc)],
                )
                _settle_spend(conn, run_id, model=model, tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost_usd)
                raise LLMParseError(f"{agent}: invalid JSON from model: {exc}") from exc

            try:
                result = model_cls(**data)
            except ValidationError as exc:
                tracing.log_call(
                    run_id, agent=agent, model=model, paper_id=paper_id, verdict="schema_invalid", attempt=attempt,
                    tokens_in=tokens_in, tokens_out=tokens_out, latency_s=latency_s,
                    cost_usd=cost_usd, reasons=[str(e) for e in exc.errors()],
                )
                _settle_spend(conn, run_id, model=model, tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost_usd)
                raise

            tracing.log_call(
                run_id, agent=agent, model=model, paper_id=paper_id, verdict="ok", attempt=attempt,
                tokens_in=tokens_in, tokens_out=tokens_out, latency_s=latency_s,
                cost_usd=cost_usd,
            )
            _settle_spend(conn, run_id, model=model, tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost_usd)
            return result
        finally:
            await _release_budget(estimated_cost)


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
    _tier_models = {
        "fast": config.OLLAMA_MODEL, "strong": config.OLLAMA_MODEL, "opus": config.OLLAMA_MODEL,
    }

    async def complete_json(
        self,
        tier: str,
        system: str,
        prompt: str,
        model_cls: type[ModelT],
        *,
        run_id: str,
        agent: str,
        agent_type: str | None = None,
        paper_id: str | None = None,
        attempt: int = 1,
        temperature: float | None = None,
        client: httpx.AsyncClient | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> ModelT:
        model = self._tier_models[tier]
        # Local model is free: reserve/settle $0.00 so a shared budget ledger
        # still makes sense across providers, but this never itself trips the
        # cap (and never needs `agent_type`'s max_tokens -- `axiom/llm.py`'s
        # wrapper doesn't take one).
        await _reserve_budget(conn, run_id, 0.0)
        try:
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
                    run_id, agent=agent, model=model, paper_id=paper_id, verdict="transport_error",
                    attempt=attempt, tokens_in=tokens_in, latency_s=latency_s,
                    reasons=[str(last_exc)],
                )
                raise LLMTransportError(f"{agent}: Ollama unreachable after retries") from last_exc

            tokens_out = _estimate_tokens(json.dumps(data))
            try:
                result = model_cls(**data)
            except ValidationError as exc:
                tracing.log_call(
                    run_id, agent=agent, model=model, paper_id=paper_id, verdict="schema_invalid", attempt=attempt,
                    tokens_in=tokens_in, tokens_out=tokens_out, latency_s=latency_s,
                    reasons=[str(e) for e in exc.errors()],
                )
                raise

            tracing.log_call(
                run_id, agent=agent, model=model, paper_id=paper_id, verdict="ok", attempt=attempt,
                tokens_in=tokens_in, tokens_out=tokens_out, latency_s=latency_s, cost_usd=0.0,
            )
            _settle_spend(conn, run_id, model=model, tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=0.0)
            return result
        finally:
            await _release_budget(0.0)


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
    agent_type: str | None = None,
    paper_id: str | None = None,
    attempt: int = 1,
    temperature: float | None = None,
    client: httpx.AsyncClient | None = None,
    conn: sqlite3.Connection | None = None,
) -> ModelT:
    """Provider-agnostic structured completion. Raises `LLMError` subclasses
    on transport/HTTP/refusal/parse failure, `pydantic.ValidationError` when
    the model's JSON doesn't satisfy `model_cls` (the caller's retry loop
    builds `VerifierError`s from `exc.errors()`, per R4). `temperature` is
    optional (coordinator/extraction leave it at the provider's default;
    synthesis's hypothesis generator passes a low value per OD16's
    generator-temperature rule). `paper_id` is forwarded to the trace line
    only (R10); omit for calls not scoped to one paper (e.g. synthesis).

    `agent_type` (e.g. "route"/"extraction"/"synthesis") selects the
    `config.AGENT_MAX_TOKENS` entry for this call's `max_tokens` (C4);
    omit for a call with no listed entry (falls back to
    `config._DEFAULT_MAX_TOKENS`).

    `client`/`conn` are the run-scoping seams for `axiom/agents/pipeline.py`
    (C2/C3): pass an `httpx.AsyncClient` to reuse one pooled client across
    every call in a run instead of opening/closing one per call, and/or a
    sqlite3 connection so the pre-call budget check reads the durable,
    cross-run `db.total_llm_spend` instead of only this process's in-memory
    total. Both default to today's per-call/in-process behavior when
    omitted.
    """
    provider = _get_provider()
    return await provider.complete_json(
        tier, system, prompt, model_cls, run_id=run_id, agent=agent, agent_type=agent_type,
        paper_id=paper_id, attempt=attempt, temperature=temperature, client=client, conn=conn,
    )
