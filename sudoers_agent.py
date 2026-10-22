"""
Sudoers Configurator Agent (Agent #25) - LLM loop and public interface.

INTERFACE (what the Base Shell team codes against)
--------------------------------------------------
    AGENT_ID, AGENT_NAME, ROUTER_DESCRIPTION   -> metadata for the router's 35-agent menu
    SudoersAgent(confirm=fn).handle(text)      -> AgentResult

    confirm(display: str) -> bool   is supplied by the shell. It receives the exact change
    the human is being asked to approve and must return True only on an explicit yes.

    AgentResult.message   text to print to the user
    AgentResult.changed   True if sudoers was actually modified
    AgentResult.actions   list of {"tool", "result"} dicts for every state-changing call
    AgentResult.error     set if the run failed (bad model, API error, step limit)

This file is an interface plus a reference loop. It is NOT wired into the class's master
shell; that code is outside this module. Requires: pip install anthropic, ANTHROPIC_API_KEY,
and root privileges (run via sudo or a root helper).

Model: defaults to MODEL below; override with SUDOERS_AGENT_MODEL.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

import sudoers_tools as st

AGENT_ID = 25
AGENT_NAME = "sudoers_configurator"
ROUTER_DESCRIPTION = (
    "Grants or revokes sudo privileges for a user or group, limited to specific commands, "
    "optionally without a password. Use for requests like 'let alice restart nginx without a "
    "password' or 'remove bob's sudo access to apt'."
)

DEFAULT_MODEL = "claude-sonnet-5-5"
MODEL = os.environ.get("SUDOERS_AGENT_MODEL", DEFAULT_MODEL)

SYSTEM_PROMPT = """You are the Sudoers Configurator Agent inside a Linux natural-language shell.
You translate a request about sudo privileges into structured tool calls. You never write
sudoers syntax yourself and you never edit files directly.

Rules:
1. Grant the LEAST privilege that satisfies the request. "restart the web server" means exactly
   `systemctl restart <that one service>`, not all of systemctl, not start/stop, not ALL.
2. Resolve vague terms with detect_services and absolute paths with resolve_command. If several
   services match (apache2 and nginx), ask the user which one instead of guessing.
3. Use nopasswd=true only if the user said no password. Otherwise leave it false.
4. Call preview_sudo_rule first. To install or remove, call add_sudo_rule / remove_sudo_rule with
   apply=true; the shell then asks the human to approve that exact change. You cannot approve it
   yourself, and if the human declines, stop and say nothing was changed.
5. If a tool returns an error, explain it plainly. Do not retry with a looser rule to get around it.
6. Finish with one or two sentences saying exactly what privilege now exists (or why none was granted).
"""

APPLY_TOOLS = ("add_sudo_rule", "remove_sudo_rule")


@dataclass
class AgentResult:
    message: str
    changed: bool = False
    actions: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def __str__(self) -> str:
        return self.message


class SudoersAgent:
    def __init__(self, confirm: Callable[[str], bool] | None = None, client=None,
                 max_turns: int = 8, model: str | None = None):
        if client is None:
            import anthropic  # lazy: the tool layer and tests work without the SDK
            client = anthropic.Anthropic()
        self.client = client
        self.confirm = confirm or self._cli_confirm
        self.max_turns = max_turns
        self.model = model or MODEL

    @staticmethod
    def _cli_confirm(display: str) -> bool:
        return input(f"\n{display}\nProceed? [y/N] ").strip().lower() == "y"

    def _create(self, messages):
        try:
            return self.client.messages.create(
                model=self.model, max_tokens=1024, system=SYSTEM_PROMPT,
                tools=st.TOOL_SCHEMAS, messages=messages,
            )
        except Exception as e:  # SDK error classes are only importable when the SDK is installed
            if type(e).__name__ in ("NotFoundError", "PermissionDeniedError"):
                raise RuntimeError(
                    f"Model {self.model!r} is not available to this API key. "
                    "Set SUDOERS_AGENT_MODEL to a model your account can use."
                ) from e
            if type(e).__name__ == "AuthenticationError":
                raise RuntimeError("Anthropic API key missing or invalid (ANTHROPIC_API_KEY).") from e
            raise

    def handle(self, user_prompt: str) -> AgentResult:
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]
        result = AgentResult(message="")

        for _ in range(self.max_turns):
            try:
                resp = self._create(messages)
            except RuntimeError as e:
                result.message, result.error = f"Could not run the agent: {e}", str(e)
                return result
            messages.append({"role": "assistant", "content": resp.content})

            if resp.stop_reason != "tool_use":
                result.message = "".join(b.text for b in resp.content if b.type == "text")
                return result

            tool_results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                args = dict(block.input)
                token = None

                if block.name in APPLY_TOOLS and args.get("apply"):
                    # Approval is computed from the EXACT arguments about to run, shown to the
                    # human, and only then turned into a token the model never sees or controls.
                    try:
                        ask = st.approval_text(block.name, args)
                    except st.SudoersError as e:
                        tool_results.append(self._tr(block.id, {"error": str(e)}))
                        continue
                    if not self.confirm(ask["display"]):
                        tool_results.append(self._tr(
                            block.id, {"error": "The user declined. Nothing was changed."}))
                        continue
                    token = st.issue_approval(ask["text"])

                out = st.dispatch(block.name, args, approval_token=token)
                if block.name in APPLY_TOOLS and args.get("apply"):
                    result.actions.append({"tool": block.name, "result": out})
                    if out.get("applied"):
                        result.changed = True
                tool_results.append(self._tr(block.id, out))
            messages.append({"role": "user", "content": tool_results})

        result.message = "Stopped: too many steps without finishing."
        result.error = "max_turns"
        return result

    @staticmethod
    def _tr(tool_use_id: str, out: dict[str, Any]) -> dict[str, Any]:
        return {"type": "tool_result", "tool_use_id": tool_use_id,
                "content": str(out), "is_error": "error" in out}


if __name__ == "__main__":
    import sys
    print(SudoersAgent().handle(" ".join(sys.argv[1:]) or input("sudoers> ")))
