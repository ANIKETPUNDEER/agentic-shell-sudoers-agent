"""
Gemini adapter for Agent 25 (optional add-on; no existing file is modified).

SudoersAgent talks to a client shaped like the Anthropic SDK:
    client.messages.create(model=, max_tokens=, system=, tools=, messages=) -> response
where response.content is a list of blocks (type "text" / "tool_use") and response.stop_reason is
"tool_use" or "end_turn". This class exposes exactly that interface on top of Google's `google-genai`
SDK, so the agent loop, the approval gate and the whole safety layer run unchanged.

    pip install google-genai
    export GEMINI_API_KEY=...            # free key from Google AI Studio
    sudo -E python3 run_gemini.py "Let alice restart nginx without a password"

Design notes
  * The model's raw Gemini reply is kept on `response.content.raw_content` and sent back verbatim on
    the next turn. Gemini requires thought signatures on function calls to be returned untouched, and
    re-building the history by hand would drop them.
  * API failures are mapped to the error names the agent loop already understands, or to RuntimeError
    (which the loop turns into a clean AgentResult.error instead of a traceback).
"""
from __future__ import annotations

import os
import uuid
from types import SimpleNamespace
from typing import Any

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"  # override with SUDOERS_AGENT_MODEL


# Names matter: SudoersAgent._create() recognises these class names.
class NotFoundError(Exception):
    pass


class PermissionDeniedError(Exception):
    pass


class AuthenticationError(Exception):
    pass


class _Blocks(list):
    """List of Anthropic-style content blocks that also remembers Gemini's original Content."""
    raw_content: Any = None


def _block_text(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _block_tool(call_id: str, name: str, args: dict) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=args)


class GeminiClient:
    default_model = DEFAULT_GEMINI_MODEL

    def __init__(self, api_key: str | None = None, genai_client: Any = None):
        from google import genai  # lazy import: the rest of the project needs no Google SDK
        from google.genai import types

        self._genai, self._types = genai, types
        if genai_client is not None:       # tests inject a fake
            self._client = genai_client
        else:
            key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise RuntimeError("Set GEMINI_API_KEY (free key from Google AI Studio) and retry.")
            self._client = genai.Client(api_key=key)
        self.messages = self  # so client.messages.create(...) works

    # ------------------------------------------------------------------ conversion
    def _tools(self, anthropic_tools: list[dict]) -> list[Any]:
        t = self._types
        decls = [
            t.FunctionDeclaration(
                name=tool["name"],
                description=tool.get("description", ""),
                parameters_json_schema=tool["input_schema"],
            )
            for tool in anthropic_tools
        ]
        return [t.Tool(function_declarations=decls)]

    def _contents(self, messages: list[dict]) -> list[Any]:
        t = self._types
        # tool_use_id -> tool name, so results can be labelled for Gemini
        names: dict[str, str] = {}
        for m in messages:
            if m["role"] == "assistant" and not isinstance(m["content"], str):
                for b in m["content"]:
                    if getattr(b, "type", None) == "tool_use":
                        names[b.id] = b.name

        out: list[Any] = []
        for m in messages:
            c = m["content"]
            if m["role"] == "assistant":
                raw = getattr(c, "raw_content", None)
                if raw is not None:
                    out.append(raw)  # verbatim, thought signatures intact
                else:
                    parts = [t.Part(text=b.text) if b.type == "text"
                             else t.Part(function_call=t.FunctionCall(id=b.id, name=b.name, args=b.input))
                             for b in c]
                    out.append(t.Content(role="model", parts=parts))
            elif isinstance(c, str):
                out.append(t.Content(role="user", parts=[t.Part(text=c)]))
            else:
                parts = []
                for r in c:
                    if r.get("type") != "tool_result":
                        continue
                    key = "error" if r.get("is_error") else "result"
                    parts.append(t.Part.from_function_response(
                        name=names.get(r["tool_use_id"], "unknown_tool"),
                        response={key: r["content"]},
                    ))
                out.append(t.Content(role="user", parts=parts))
        return out

    def _to_anthropic(self, resp: Any) -> SimpleNamespace:
        cands = getattr(resp, "candidates", None) or []
        if not cands or cands[0].content is None or not cands[0].content.parts:
            reason = getattr(cands[0], "finish_reason", None) if cands else None
            raise RuntimeError(f"Gemini returned no usable content (finish reason: {reason}). "
                               "The request may have been blocked or the model is overloaded; try again.")
        content = cands[0].content
        blocks = _Blocks()
        blocks.raw_content = content
        calls = 0
        for p in content.parts:
            fc = getattr(p, "function_call", None)
            if fc is not None and fc.name:
                calls += 1
                blocks.append(_block_tool(fc.id or f"call_{uuid.uuid4().hex[:12]}", fc.name, dict(fc.args or {})))
            elif getattr(p, "text", None) and not getattr(p, "thought", False):
                blocks.append(_block_text(p.text))
        return SimpleNamespace(content=blocks, stop_reason="tool_use" if calls else "end_turn")

    # ------------------------------------------------------------------ the call
    def create(self, *, model: str, max_tokens: int, system: str, tools: list[dict],
               messages: list[dict]) -> SimpleNamespace:
        t = self._types
        config = t.GenerateContentConfig(
            system_instruction=system,
            tools=self._tools(tools),
            max_output_tokens=max(max_tokens, 1024),
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),
        )
        try:
            resp = self._client.models.generate_content(
                model=model, contents=self._contents(messages), config=config)
        except Exception as e:
            self._raise_mapped(e)
        return self._to_anthropic(resp)

    @staticmethod
    def _raise_mapped(e: Exception) -> None:
        code = getattr(e, "code", None)
        msg = str(getattr(e, "message", None) or e)
        if code is None:
            raise e  # not an API error (programming bug): let it surface
        low = msg.lower()
        if code in (400, 401, 403) and ("api key" in low or "api_key" in low):
            raise AuthenticationError(msg) from e
        if code == 404:
            raise NotFoundError(msg) from e
        if code == 403:
            raise PermissionDeniedError(msg) from e
        if code == 429:
            raise RuntimeError("Gemini rate limit or free-tier quota reached. Wait a minute and retry.") from e
        raise RuntimeError(f"Gemini API error {code}: {msg}") from e
