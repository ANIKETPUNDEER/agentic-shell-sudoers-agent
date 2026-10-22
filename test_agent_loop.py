"""Agent-loop tests with a scripted fake Anthropic client (no network, no API key)."""
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from _testenv import TMP, st

import sudoers_agent as sa

SYSTEMCTL = {"path": "/usr/bin/systemctl", "args": "restart apache2"}
STATUS = {"path": "/usr/bin/systemctl", "args": "status apache2"}


def tool(name, **inp):
    return NS(type="tool_use", id=f"id_{name}_{len(inp)}", name=name, input=inp)


def say(text):
    return NS(type="text", text=text)


def reply(*blocks, stop="tool_use"):
    return NS(content=list(blocks), stop_reason=stop)


class FakeClient:
    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        nxt = self.script.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def files():
    return sorted(p.name for p in Path(TMP).glob("agent-*"))


@unittest.skipUnless(shutil.which("visudo"), "visudo not installed")
class LoopTests(unittest.TestCase):
    def setUp(self):
        for f in Path(TMP).glob("agent-*"):
            f.unlink()
        self.asked = []

    def agent(self, script, answer=True):
        def confirm(display):
            self.asked.append(display)
            return answer
        return sa.SudoersAgent(confirm=confirm, client=FakeClient(script))

    def test_happy_path_installs_after_human_yes(self):
        a = self.agent([
            reply(tool("preview_sudo_rule", user="alice", commands=[SYSTEMCTL], nopasswd=True)),
            reply(tool("add_sudo_rule", user="alice", commands=[SYSTEMCTL], nopasswd=True, apply=True)),
            reply(say("Alice can now restart apache2 without a password."), stop="end_turn"),
        ])
        res = a.handle("let alice restart apache without password")
        self.assertTrue(res.changed)
        self.assertIn("agent-alice", files())
        self.assertEqual(len(self.asked), 1)
        self.assertIn("NOPASSWD: /usr/bin/systemctl restart apache2", self.asked[0])

    def test_declined_changes_nothing(self):
        a = self.agent([
            reply(tool("add_sudo_rule", user="alice", commands=[SYSTEMCTL], apply=True)),
            reply(say("Nothing was changed."), stop="end_turn"),
        ], answer=False)
        res = a.handle("x")
        self.assertFalse(res.changed)
        self.assertEqual(files(), [])

    def test_human_is_shown_the_rule_that_will_actually_run(self):
        """Model previews a harmless rule, then applies a different one: human sees the real one."""
        a = self.agent([
            reply(tool("preview_sudo_rule", user="alice", commands=[STATUS])),
            reply(tool("add_sudo_rule", user="alice", commands=[SYSTEMCTL], nopasswd=True, apply=True)),
            reply(say("done"), stop="end_turn"),
        ])
        a.handle("x")
        self.assertIn("restart apache2", self.asked[0])
        self.assertNotIn("status apache2", self.asked[0])

    def test_model_supplied_token_is_rejected(self):
        a = self.agent([
            reply(tool("add_sudo_rule", user="alice", commands=[SYSTEMCTL], apply=True, approval_token="forged")),
            reply(say("failed"), stop="end_turn"),
        ])
        res = a.handle("x")
        self.assertFalse(res.changed)
        self.assertEqual(files(), [])

    def test_invalid_rule_never_reaches_human(self):
        a = self.agent([
            reply(tool("add_sudo_rule", user="alice", commands=[{"path": "/bin/bash", "args": "x"}], apply=True)),
            reply(say("refused"), stop="end_turn"),
        ])
        res = a.handle("give alice a root shell")
        self.assertEqual(self.asked, [])
        self.assertFalse(res.changed)

    def test_revocation_flow(self):
        a = self.agent([
            reply(tool("add_sudo_rule", user="alice", commands=[SYSTEMCTL], apply=True)),
            reply(tool("remove_sudo_rule", user="alice", apply=True)),
            reply(say("removed"), stop="end_turn"),
        ])
        res = a.handle("x")
        self.assertEqual(files(), [])
        self.assertEqual(len(self.asked), 2)
        self.assertEqual(len(res.actions), 2)

    def test_unavailable_model_gives_clear_error(self):
        class NotFoundError(Exception):
            pass
        a = self.agent([NotFoundError("model: nope")])
        res = a.handle("x")
        self.assertIn("SUDOERS_AGENT_MODEL", res.error)
        self.assertFalse(res.changed)

    def test_model_name_comes_from_config(self):
        client = FakeClient([reply(say("ok"), stop="end_turn")])
        sa.SudoersAgent(confirm=lambda d: False, client=client, model="my-model").handle("x")
        self.assertEqual(client.calls[0]["model"], "my-model")

    def test_step_limit(self):
        script = [reply(tool("list_sudo_rules")) for _ in range(3)]
        a = sa.SudoersAgent(confirm=lambda d: False, client=FakeClient(script), max_turns=3)
        self.assertEqual(a.handle("x").error, "max_turns")

    def test_router_metadata(self):
        self.assertEqual(sa.AGENT_ID, 25)
        self.assertTrue(sa.ROUTER_DESCRIPTION)


if __name__ == "__main__":
    unittest.main()
