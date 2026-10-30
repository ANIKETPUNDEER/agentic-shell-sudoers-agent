"""Offline tests for the Gemini adapter: real google-genai types, fake network. Skipped if the SDK is absent."""
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from _testenv import TMP, st

try:
    from google.genai import types
    import gemini_client as gc
    import sudoers_agent as sa
    HAVE_SDK = True
except ImportError:  # pragma: no cover
    HAVE_SDK = False

SYSTEMCTL = {"path": "/usr/bin/systemctl", "args": "restart apache2"}


def model_turn(*parts):
    return NS(candidates=[NS(content=types.Content(role="model", parts=list(parts)), finish_reason="STOP")])


def call(name, **args):
    return types.Part(function_call=types.FunctionCall(id=f"id-{name}", name=name, args=args))


class FakeModels:
    def __init__(self, script):
        self.script, self.requests = list(script), []

    def generate_content(self, *, model, contents, config):
        self.requests.append((model, contents, config))
        nxt = self.script.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def make(script):
    fake = NS(models=FakeModels(script))
    return gc.GeminiClient(genai_client=fake), fake.models


@unittest.skipUnless(HAVE_SDK and shutil.which("visudo"), "needs google-genai and visudo")
class GeminiAdapterTests(unittest.TestCase):
    def setUp(self):
        for f in Path(TMP).glob("agent-*"):
            f.unlink()

    def test_tool_schemas_convert_without_error(self):
        client, _ = make([])
        tools = client._tools(st.TOOL_SCHEMAS)
        names = [d.name for d in tools[0].function_declarations]
        self.assertEqual(names, [t["name"] for t in st.TOOL_SCHEMAS])
        self.assertIsNotNone(tools[0].function_declarations[0].parameters_json_schema)

    def test_response_conversion(self):
        client, _ = make([])
        r = client._to_anthropic(model_turn(types.Part(text="hi"), call("list_sudo_rules")))
        self.assertEqual(r.stop_reason, "tool_use")
        self.assertEqual([b.type for b in r.content], ["text", "tool_use"])
        self.assertEqual(r.content[1].name, "list_sudo_rules")
        r2 = client._to_anthropic(model_turn(types.Part(text="done")))
        self.assertEqual(r2.stop_reason, "end_turn")

    def test_empty_or_blocked_response_is_clean_error(self):
        client, _ = make([])
        with self.assertRaises(RuntimeError):
            client._to_anthropic(NS(candidates=[]))

    def test_full_flow_through_agent_loop_installs_after_yes(self):
        client, models = make([
            model_turn(call("add_sudo_rule", user="alice", commands=[SYSTEMCTL], nopasswd=True, apply=True)),
            model_turn(types.Part(text="Alice can restart apache2 without a password.")),
        ])
        asked = []
        agent = sa.SudoersAgent(confirm=lambda d: asked.append(d) or True, client=client, model="gemini-test")
        res = agent.handle("let alice restart apache without a password")
        self.assertTrue(res.changed)
        self.assertTrue((Path(TMP) / "agent-alice").exists())
        self.assertIn("NOPASSWD: /usr/bin/systemctl restart apache2", asked[0])
        # second request carried the model turn back verbatim + a function response
        contents = models.requests[1][1]
        self.assertEqual([c.role for c in contents], ["user", "model", "user"])
        fr = contents[2].parts[0].function_response
        self.assertEqual(fr.name, "add_sudo_rule")
        self.assertIn("result", fr.response)
        self.assertEqual(models.requests[1][0], "gemini-test")
        self.assertTrue(models.requests[0][2].automatic_function_calling.disable)

    def test_decline_changes_nothing(self):
        client, _ = make([
            model_turn(call("add_sudo_rule", user="alice", commands=[SYSTEMCTL], apply=True)),
            model_turn(types.Part(text="Nothing changed.")),
        ])
        res = sa.SudoersAgent(confirm=lambda d: False, client=client, model="m").handle("x")
        self.assertFalse(res.changed)
        self.assertFalse((Path(TMP) / "agent-alice").exists())

    def test_tool_error_is_labelled_error(self):
        client, models = make([
            model_turn(call("add_sudo_rule", user="alice", commands=[{"path": "/bin/bash"}], apply=True)),
            model_turn(types.Part(text="Refused.")),
        ])
        sa.SudoersAgent(confirm=lambda d: True, client=client, model="m").handle("root shell please")
        fr = models.requests[1][1][2].parts[0].function_response
        self.assertIn("error", fr.response)

    def test_api_errors_map_to_clean_messages(self):
        def err(code, msg):
            e = Exception(msg)
            e.code, e.message = code, msg
            return e
        cases = {
            err(400, "API key not valid. Please pass a valid API key."): "API key",
            err(404, "models/foo is not found"): "SUDOERS_AGENT_MODEL",
            err(429, "quota"): "rate limit",
            err(500, "boom"): "Gemini API error 500",
        }
        for exc, expect in cases.items():
            client, _ = make([exc])
            res = sa.SudoersAgent(confirm=lambda d: False, client=client, model="m").handle("x")
            self.assertIn(expect.lower(), (res.error or "").lower(), msg=str(exc))
            self.assertFalse(res.changed)

    def test_missing_key_message(self):
        import os
        saved = {k: os.environ.pop(k, None) for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
        try:
            with self.assertRaises(RuntimeError):
                gc.GeminiClient()
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v


if __name__ == "__main__":
    unittest.main()
