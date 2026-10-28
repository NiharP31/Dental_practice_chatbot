"""LLM providers behind one call: complete(system, history, tools) -> assistant message.

Chain comes from LLM_CHAIN (default "claude,gemini"); providers without a key are skipped, and the
caller falls back to a canned reply when every provider fails. History uses a provider-neutral
format so a turn can switch provider mid-way (e.g. Claude makes a tool call, then Gemini finishes):
  {"role": "user", "text": str}
  {"role": "assistant", "text": str, "tool_calls": [{"id", "name", "args", "sig"?}], "provider"?: str}
  {"role": "tool", "id": str, "name": str, "result": dict}
"""
import base64
import json
import logging
import re
import threading
import time
import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import config
from app.db import LlmUsage

log = logging.getLogger("llm")
TIMEOUT_S = 25
MAX_RETRY_WAIT_S = 30  # a longer suggested wait means: move on to the next provider instead
DEFAULT_RETRY_WAIT_S = 2
# Gemini 3 rejects a current-turn function call without a thought signature (400 "Function call is
# missing a thought_signature"). Calls Claude made have none; Gemini accepts this placeholder for them.
FOREIGN_CALL_SIGNATURE = b"skip_thought_signature_validator"

# USD per 1M tokens (input, output) for prompts <= 100K tokens. Cache reads 0.1x input, 5-min writes 1.25x.
CLAUDE_PRICES = {"claude-haiku-5-5": (0.10, 0.50)}
UNKNOWN_CLAUDE_PRICE = (5.00, 25.00)  # budget conservatively for a model we have no price for


class LLMUnavailable(Exception):
    pass


_throttle_lock, _last_call = threading.Lock(), [0.0]


def throttle() -> None:
    """Space calls at least 60/LLM_MAX_RPM seconds apart (free-tier Gemini evals; 0 = off)."""
    if config.LLM_MAX_RPM <= 0:
        return
    with _throttle_lock:
        wait = _last_call[0] + 60 / config.LLM_MAX_RPM - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.monotonic()


def suggested_delay(err) -> float:
    """Seconds from a Gemini 429's RetryInfo ("retryDelay": "17s"), or a short default."""
    m = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s", str(getattr(err, "details", "")) + str(err))
    return float(m[1]) if m else DEFAULT_RETRY_WAIT_S


def _quota(err) -> str:
    m = re.search(r"quotaId['\"]?\s*:\s*['\"]([^'\"]+).{0,200}?quotaValue['\"]?\s*:\s*['\"]?(\d+)", str(getattr(err, "details", "")))
    return f"{m[1]}={m[2]}" if m else "unknown"


def _tool_id(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", raw)[:64] or "call"  # Claude requires ^[A-Za-z0-9_-]+$


class Claude:
    billed = True

    def __init__(self, model: str):
        import anthropic
        self.model, self.name = model, f"claude:{model}"
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY, timeout=TIMEOUT_S, max_retries=1)

    def complete(self, system: str, history: list[dict], tools: list[dict]) -> dict:
        throttle()
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            # Thinking off: the guard rewrites stored replies and turns can switch provider, so history
            # must stay editable; Claude 5.5 thinking blocks are invalidated (400) by any history edit.
            thinking={"type": "disabled"},
            output_config={"effort": "low"},
            # Render order is tools -> system -> messages, so one breakpoint here caches both. The
            # system prompt is static (the clock lives in each user turn) so the cache actually hits.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            tools=[{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools],
            messages=self._messages(history),
        )
        if resp.stop_reason in ("refusal", "max_tokens"):
            raise LLMUnavailable(f"claude stop_reason={resp.stop_reason}")
        text, calls = [], []
        for b in resp.content:
            if b.type == "text":
                text.append(b.text)
            elif b.type == "tool_use":
                calls.append({"id": b.id, "name": b.name, "args": dict(b.input)})
        u = resp.usage
        usage = {"input": u.input_tokens, "output": u.output_tokens,
                 "cache_read": u.cache_read_input_tokens or 0, "cache_write": u.cache_creation_input_tokens or 0}
        pin, pout = CLAUDE_PRICES.get(self.model, UNKNOWN_CLAUDE_PRICE)
        cost = (usage["input"] * pin + usage["cache_read"] * pin * 0.1 + usage["cache_write"] * pin * 1.25
                + usage["output"] * pout) / 1e6
        return {"role": "assistant", "text": "".join(text), "tool_calls": calls, "usage": usage, "cost_usd": cost}

    @staticmethod
    def _messages(history: list[dict]) -> list[dict]:
        out = []
        for m in history:
            if m["role"] == "user":
                out.append({"role": "user", "content": [{"type": "text", "text": m["text"]}]})
            elif m["role"] == "assistant":
                blocks = [{"type": "text", "text": m["text"]}] if m.get("text") else []
                blocks += [{"type": "tool_use", "id": _tool_id(c["id"]), "name": c["name"], "input": c["args"]}
                           for c in m.get("tool_calls", [])]
                if blocks:
                    out.append({"role": "assistant", "content": blocks})
            else:  # all results for one assistant turn go back in a single user message
                block = {"type": "tool_result", "tool_use_id": _tool_id(m["id"]),
                         "content": json.dumps(m["result"], default=str),
                         **({"is_error": True} if "error" in m["result"] else {})}
                if out and out[-1]["role"] == "user" and out[-1]["content"][0]["type"] == "tool_result":
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return out


class Gemini:
    billed = False  # free tier

    def __init__(self, model: str):
        from google import genai
        from google.genai import errors, types
        self.types, self.errors, self.model, self.name = types, errors, model, f"gemini:{model}"
        # SDK retries off: we decide (one 429 retry, then the next provider) so a patient never waits minutes.
        self.client = genai.Client(api_key=config.GEMINI_API_KEY, http_options=types.HttpOptions(
            timeout=TIMEOUT_S * 1000, retry_options=types.HttpRetryOptions(attempts=1)))
        self.skip_until = 0.0  # monotonic time; set when a 429 asks us to wait longer than we'll block

    def _generate(self, **kw):
        if time.monotonic() < self.skip_until:
            raise LLMUnavailable(f"{self.model} rate-limited for {self.skip_until - time.monotonic():.0f}s more")
        for attempt in (1, 2):
            throttle()
            try:
                return self.client.models.generate_content(model=self.model, **kw)
            except self.errors.ClientError as e:
                delay = suggested_delay(e)
                if e.code == 429:
                    log.warning("rate_limited model=%s quota=%s retry_in=%.0fs", self.model, _quota(e), delay)
                    if delay > MAX_RETRY_WAIT_S:
                        # e.g. the free tier's daily cap: don't spend a call (and a throttle slot) per turn on it
                        self.skip_until = time.monotonic() + delay
                if e.code != 429 or attempt == 2 or delay > MAX_RETRY_WAIT_S:
                    raise
                time.sleep(delay)

    def complete(self, system: str, history: list[dict], tools: list[dict]) -> dict:
        t = self.types
        cfg = t.GenerateContentConfig(
            system_instruction=system,
            tools=[t.Tool(function_declarations=[
                t.FunctionDeclaration(name=x["name"], description=x["description"], parameters_json_schema=x["parameters"])
                for x in tools])],
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),
        )
        resp = self._generate(contents=self._contents(history), config=cfg)
        if not resp.candidates or not resp.candidates[0].content or not resp.candidates[0].content.parts:
            raise LLMUnavailable(f"empty Gemini response: {getattr(resp, 'prompt_feedback', None)}")
        text, calls = [], []
        for part in resp.candidates[0].content.parts:
            if part.function_call:
                fc = part.function_call
                calls.append({"id": fc.id or f"call_{uuid.uuid4().hex[:12]}", "name": fc.name, "args": dict(fc.args or {}),
                              # Gemini 3 rejects a turn whose function calls come back without their signature.
                              "sig": base64.b64encode(part.thought_signature).decode() if part.thought_signature else None})
            elif part.text and not part.thought:
                text.append(part.text)
        um = resp.usage_metadata
        usage = {"input": getattr(um, "prompt_token_count", 0) or 0, "output": getattr(um, "candidates_token_count", 0) or 0,
                 "cache_read": getattr(um, "cached_content_token_count", 0) or 0, "cache_write": 0}
        return {"role": "assistant", "text": "".join(text), "tool_calls": calls, "usage": usage, "cost_usd": 0.0}

    def _contents(self, history):
        t, out = self.types, []
        for m in history:
            if m["role"] == "user":
                out.append(t.Content(role="user", parts=[t.Part(text=m["text"])]))
            elif m["role"] == "assistant":
                parts = [t.Part(text=m["text"])] if m.get("text") else []
                parts += [t.Part(function_call=t.FunctionCall(id=c["id"], name=c["name"], args=c["args"]),
                                 thought_signature=base64.b64decode(c["sig"]) if c.get("sig") else FOREIGN_CALL_SIGNATURE)
                          for c in m.get("tool_calls", [])]
                out.append(t.Content(role="model", parts=parts or [t.Part(text=" ")]))
            else:  # consecutive tool results go back together in one turn
                part = t.Part(function_response=t.FunctionResponse(id=m["id"], name=m["name"], response=m["result"]))
                if out and out[-1].role == "user" and out[-1].parts[0].function_response:
                    out[-1].parts.append(part)
                else:
                    out.append(t.Content(role="user", parts=[part]))
        return out


def providers() -> list:
    out = []
    for name in (n.strip().lower() for n in config.LLM_CHAIN.split(",") if n.strip()):
        if name == "claude":
            if config.ANTHROPIC_API_KEY:
                out.append(Claude(config.ANTHROPIC_MODEL))
        elif name == "gemini":
            if config.GEMINI_API_KEY and config.GEMINI_MODEL:
                out.append(Gemini(config.GEMINI_MODEL))
        else:
            log.warning("unknown provider %r in LLM_CHAIN", name)
    log.info("llm_chain %s", " -> ".join(p.name for p in out) or "(none: canned replies only)")
    return out


_PROVIDERS = None
_budget_logged_day = [None]


def spent_today(db: Session) -> float:
    return float(db.scalar(select(func.coalesce(func.sum(LlmUsage.cost_usd), 0.0))
                           .where(LlmUsage.day == config.office_now().date())))


def _over_budget(db: Session) -> bool:
    spent = spent_today(db)
    if spent < config.DAILY_LLM_BUDGET_USD:
        return False
    today = config.office_now().date()
    if _budget_logged_day[0] != today:
        _budget_logged_day[0] = today
        log.warning("llm_budget_exhausted spent=$%.4f budget=$%.2f: billed providers skipped until tomorrow",
                    spent, config.DAILY_LLM_BUDGET_USD)
    return True


def _record(db: Session, session_id: str, provider: str, msg: dict) -> None:
    u = msg.get("usage") or {}
    try:
        db.add(LlmUsage(day=config.office_now().date(), session_id=session_id, provider=provider,
                        input_tokens=u.get("input", 0), output_tokens=u.get("output", 0),
                        cache_read_tokens=u.get("cache_read", 0), cache_write_tokens=u.get("cache_write", 0),
                        cost_usd=msg.get("cost_usd", 0.0)))
        db.commit()
    except Exception:  # spend tracking must never cost the patient their reply
        db.rollback()
        log.exception("usage_record_failed provider=%s", provider)


def complete(system: str, history: list[dict], tools: list[dict], db: Session | None = None,
             session_id: str = "") -> dict:
    """Try each provider in order; record token usage. Raises LLMUnavailable if all fail."""
    global _PROVIDERS
    if _PROVIDERS is None:
        _PROVIDERS = providers()
    for p in _PROVIDERS:
        if p.billed and db is not None and _over_budget(db):
            continue
        try:
            msg = p.complete(system, history, tools)
        except Exception as e:  # any provider failure (quota, timeout, 5xx, refusal, bad response) -> next one
            log.warning("llm_failed provider=%s error=%s", p.name, type(e).__name__ + ": " + str(e)[:200])
            continue
        msg["provider"] = p.name
        if db is not None:
            _record(db, session_id, p.name, msg)
        return msg
    raise LLMUnavailable("no LLM provider succeeded")
