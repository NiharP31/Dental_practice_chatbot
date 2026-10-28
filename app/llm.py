"""LLM providers behind one call: complete(system, history, tools) -> assistant message.

History uses a provider-neutral format so a turn can fall back from Gemini to DeepSeek mid-way:
  {"role": "user", "text": str}
  {"role": "assistant", "text": str, "tool_calls": [{"id", "name", "args", "sig"?}]}
  {"role": "tool", "id": str, "name": str, "result": dict}
"""
import base64
import json
import logging
import uuid

from app import config

log = logging.getLogger("llm")
TIMEOUT_S = 25


class LLMUnavailable(Exception):
    pass


class Gemini:
    name = "gemini"

    def __init__(self):
        from google import genai
        from google.genai import types
        self.types = types
        self.client = genai.Client(api_key=config.GEMINI_API_KEY,
                                   http_options=types.HttpOptions(timeout=TIMEOUT_S * 1000))

    def complete(self, system: str, history: list[dict], tools: list[dict]) -> dict:
        t = self.types
        cfg = t.GenerateContentConfig(
            system_instruction=system,
            tools=[t.Tool(function_declarations=[
                t.FunctionDeclaration(name=x["name"], description=x["description"], parameters_json_schema=x["parameters"])
                for x in tools])],
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),
        )
        resp = self.client.models.generate_content(model=config.GEMINI_MODEL, contents=self._contents(history), config=cfg)
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
        return {"role": "assistant", "text": "".join(text), "tool_calls": calls}

    def _contents(self, history):
        t, out = self.types, []
        for m in history:
            if m["role"] == "user":
                out.append(t.Content(role="user", parts=[t.Part(text=m["text"])]))
            elif m["role"] == "assistant":
                parts = [t.Part(text=m["text"])] if m.get("text") else []
                parts += [t.Part(function_call=t.FunctionCall(id=c["id"], name=c["name"], args=c["args"]),
                                 thought_signature=base64.b64decode(c["sig"]) if c.get("sig") else None)
                          for c in m.get("tool_calls", [])]
                out.append(t.Content(role="model", parts=parts or [t.Part(text=" ")]))
            else:  # consecutive tool results go back together in one turn
                part = t.Part(function_response=t.FunctionResponse(id=m["id"], name=m["name"], response=m["result"]))
                if out and out[-1].role == "user" and out[-1].parts[0].function_response:
                    out[-1].parts.append(part)
                else:
                    out.append(t.Content(role="user", parts=[part]))
        return out


class DeepSeek:
    name = "deepseek"

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com",
                             timeout=TIMEOUT_S, max_retries=0)

    def complete(self, system: str, history: list[dict], tools: list[dict]) -> dict:
        messages = [{"role": "system", "content": system}]
        for m in history:
            if m["role"] == "user":
                messages.append({"role": "user", "content": m["text"]})
            elif m["role"] == "assistant":
                msg = {"role": "assistant", "content": m.get("text") or None}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                          "function": {"name": c["name"], "arguments": json.dumps(c["args"])}}
                                         for c in m["tool_calls"]]
                messages.append(msg)
            else:
                messages.append({"role": "tool", "tool_call_id": m["id"], "content": json.dumps(m["result"], default=str)})
        resp = self.client.chat.completions.create(
            model=config.DEEPSEEK_MODEL, messages=messages,
            tools=[{"type": "function", "function": x} for x in tools],
            extra_body={"thinking": {"type": "disabled"}},  # thinking is on by default; too slow for chat
        )
        msg = resp.choices[0].message
        calls = []
        for c in msg.tool_calls or []:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_invalid_json": c.function.arguments}  # dispatcher rejects it; the model retries
            calls.append({"id": c.id, "name": c.function.name, "args": args})
        return {"role": "assistant", "text": msg.content or "", "tool_calls": calls}


def providers() -> list:
    out = []
    if config.GEMINI_API_KEY and config.GEMINI_MODEL:
        out.append(Gemini())
    if config.DEEPSEEK_API_KEY and config.DEEPSEEK_MODEL:
        out.append(DeepSeek())
    return out


_PROVIDERS = None


def complete(system: str, history: list[dict], tools: list[dict]) -> dict:
    """Try each configured provider in order. Raises LLMUnavailable if all fail."""
    global _PROVIDERS
    if _PROVIDERS is None:
        _PROVIDERS = providers()
    for p in _PROVIDERS:
        try:
            msg = p.complete(system, history, tools)
            msg["provider"] = p.name
            return msg
        except Exception as e:  # any provider failure (quota, timeout, 5xx, bad response) -> next one
            log.warning("llm_failed provider=%s error=%s", p.name, type(e).__name__ + ": " + str(e)[:200])
    raise LLMUnavailable("no LLM provider succeeded")
