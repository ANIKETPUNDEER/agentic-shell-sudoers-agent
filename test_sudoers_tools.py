"""Tool-layer tests. Run:  python3 -m unittest -v   (needs visudo; temp dir, never touches /etc)"""
import os
import shutil
import unittest
from pathlib import Path

from _testenv import TMP, approved, st

SYSTEMCTL = {"path": "/usr/bin/systemctl", "args": "restart apache2"}
STATUS = {"path": "/usr/bin/systemctl", "args": "status apache2"}


def files():
    return sorted(p.name for p in Path(TMP).glob("agent-*"))


@unittest.skipUnless(shutil.which("visudo"), "visudo not installed")
class SudoersTests(unittest.TestCase):
    def setUp(self):
        for f in Path(TMP).glob("agent-*"):
            f.unlink()
        os.environ.pop("SUDOERS_AGENT_ALLOWLIST", None)

    # ---- the assignment's own example ----
    def test_example_prompt_rule(self):
        r = approved("add_sudo_rule", user="alice", commands=[SYSTEMCTL], nopasswd=True)
        self.assertEqual(r["rule"], "alice ALL=(root) NOPASSWD: /usr/bin/systemctl restart apache2")
        self.assertTrue(r["applied"])
        f = Path(TMP) / "agent-alice"
        self.assertEqual(oct(f.stat().st_mode & 0o777), "0o440")

    def test_preview_writes_nothing(self):
        st.add_sudo_rule("alice", [SYSTEMCTL], nopasswd=True)
        self.assertEqual(files(), [])

    def test_idempotent(self):
        approved("add_sudo_rule", user="alice", commands=[SYSTEMCTL])
        r = approved("add_sudo_rule", user="alice", commands=[SYSTEMCTL])
        self.assertFalse(r["applied"])
        self.assertEqual(len(st.list_sudo_rules("alice")["rules"]["agent-alice"]), 1)

    def test_second_rule_appends(self):
        approved("add_sudo_rule", user="alice", commands=[SYSTEMCTL])
        approved("add_sudo_rule", user="alice", commands=[STATUS])
        self.assertEqual(len(st.list_sudo_rules("alice")["rules"]["agent-alice"]), 2)

    # ---- approval binding (review item 1) ----
    def test_apply_without_token_refused(self):
        out = st.dispatch("add_sudo_rule", {"user": "alice", "commands": [SYSTEMCTL], "apply": True})
        self.assertIn("error", out)
        self.assertEqual(files(), [])

    def test_token_for_rule_A_cannot_install_rule_B(self):
        ask = st.approval_text("add_sudo_rule", {"user": "alice", "commands": [STATUS], "apply": True})
        token = st.issue_approval(ask["text"])
        out = st.dispatch("add_sudo_rule", {"user": "alice", "commands": [SYSTEMCTL], "apply": True},
                          approval_token=token)
        self.assertIn("error", out)
        self.assertEqual(files(), [])

    def test_token_cannot_change_nopasswd(self):
        ask = st.approval_text("add_sudo_rule", {"user": "alice", "commands": [SYSTEMCTL], "apply": True})
        token = st.issue_approval(ask["text"])
        out = st.dispatch("add_sudo_rule",
                          {"user": "alice", "commands": [SYSTEMCTL], "nopasswd": True, "apply": True},
                          approval_token=token)
        self.assertIn("error", out)
        self.assertEqual(files(), [])

    def test_token_cannot_change_user(self):
        ask = st.approval_text("add_sudo_rule", {"user": "alice", "commands": [SYSTEMCTL], "apply": True})
        token = st.issue_approval(ask["text"])
        out = st.dispatch("add_sudo_rule", {"user": "bob", "commands": [SYSTEMCTL], "apply": True},
                          approval_token=token)
        self.assertIn("error", out)

    def test_model_cannot_supply_token_or_dangerous_flag(self):
        for extra in ({"approval_token": "x"}, {"allow_dangerous": True}):
            out = st.dispatch("add_sudo_rule", {"user": "alice", "commands": [SYSTEMCTL], **extra})
            self.assertIn("error", out)
        names = [p for t in st.TOOL_SCHEMAS for p in t["input_schema"].get("properties", {})]
        self.assertNotIn("allow_dangerous", names)
        self.assertNotIn("approval_token", names)

    # ---- injection / policy ----
    def test_newline_injection_rejected(self):
        with self.assertRaises(st.SudoersError):
            st.preview_sudo_rule("alice", [{"path": "/usr/bin/systemctl",
                                            "args": "restart apache2\nalice ALL=(ALL) NOPASSWD: ALL"}])

    def test_user_injection_rejected(self):
        for bad in ["alice ALL=(ALL) NOPASSWD: ALL #", "al ice", "alice,bob", "ALICE", "../x", ""]:
            with self.assertRaises(st.SudoersError, msg=bad):
                st.preview_sudo_rule(bad, [SYSTEMCTL])

    def test_comma_colon_equals_in_args_rejected(self):
        for bad in ["restart apache2, /bin/bash", "a:b", "a=b"]:
            with self.assertRaises(st.SudoersError, msg=bad):
                st.preview_sudo_rule("alice", [{"path": "/usr/bin/systemctl", "args": bad}])

    def test_wildcard_rejected(self):
        with self.assertRaises(st.SudoersError):
            st.preview_sudo_rule("alice", [{"path": "/usr/bin/systemctl", "args": "restart *"}])

    def test_all_rejected(self):
        with self.assertRaises(st.SudoersError):
            st.preview_sudo_rule("alice", [{"path": "ALL"}])

    def test_shell_escape_binaries_rejected(self):
        for p in ["/bin/bash", "/usr/bin/vim", "/usr/bin/find", "/usr/bin/python3"]:
            with self.assertRaises(st.SudoersError, msg=p):
                st.preview_sudo_rule("alice", [{"path": p, "args": "x"}])

    def test_unpinned_systemctl_rejected(self):
        with self.assertRaises(st.SudoersError):
            st.preview_sudo_rule("alice", [{"path": "/usr/bin/systemctl"}])

    def test_relative_path_rejected(self):
        with self.assertRaises(st.SudoersError):
            st.preview_sudo_rule("alice", [{"path": "systemctl", "args": "restart apache2"}])

    def test_root_rejected(self):
        with self.assertRaises(st.SudoersError):
            st.preview_sudo_rule("root", [SYSTEMCTL])

    # ---- hard allowlist (review item 2) ----
    def test_allowlist_blocks_everything_else(self):
        os.environ["SUDOERS_AGENT_ALLOWLIST"] = "systemctl"
        st.preview_sudo_rule("alice", [SYSTEMCTL])  # allowed
        with self.assertRaises(st.SudoersError):
            st.preview_sudo_rule("alice", [{"path": "/usr/bin/apt", "args": "update"}])

    # ---- groups, removal / revocation ----
    def test_group_rule(self):
        r = approved("add_sudo_rule", user="%webadmins", commands=[SYSTEMCTL])
        self.assertTrue(r["rule"].startswith("%webadmins ALL=(root) PASSWD:"))
        self.assertIn("agent-group-webadmins", files())

    def test_revocation_needs_approval_then_removes(self):
        approved("add_sudo_rule", user="alice", commands=[SYSTEMCTL])
        out = st.dispatch("remove_sudo_rule", {"user": "alice", "apply": True})
        self.assertIn("error", out)
        self.assertIn("agent-alice", files())
        approved("remove_sudo_rule", user="alice")
        self.assertEqual(files(), [])

    def test_remove_one_command_keeps_others(self):
        approved("add_sudo_rule", user="alice", commands=[SYSTEMCTL])
        approved("add_sudo_rule", user="alice", commands=[{"path": "/usr/bin/journalctl", "args": "-u apache2"}])
        approved("remove_sudo_rule", user="alice", command_path="/usr/bin/journalctl")
        left = st.list_sudo_rules("alice")["rules"]["agent-alice"]
        self.assertEqual(len(left), 1)
        self.assertIn("systemctl", left[0])

    def test_remove_preview_does_not_delete(self):
        approved("add_sudo_rule", user="alice", commands=[SYSTEMCTL])
        st.remove_sudo_rule("alice")
        self.assertIn("agent-alice", files())

    # ---- failure handling / rollback (review item 4) ----
    def test_dispatch_returns_errors_as_data(self):
        self.assertIn("error", st.dispatch("add_sudo_rule", {"user": "alice; rm -rf /", "commands": [SYSTEMCTL]}))
        self.assertIn("error", st.dispatch("nope", {}))
        self.assertIn("error", st.dispatch("add_sudo_rule", {"bogus": 1}))

    def test_no_temp_files_left_behind(self):
        approved("add_sudo_rule", user="alice", commands=[SYSTEMCTL])
        self.assertEqual([p.name for p in Path(TMP).iterdir() if p.name.endswith(".tmp")], [])

    def test_audit_log_written(self):
        approved("add_sudo_rule", user="alice", commands=[SYSTEMCTL])
        self.assertIn('"event": "add"', Path(os.environ["SUDOERS_AGENT_LOG"]).read_text())

    def test_visudo_rejection_leaves_state_unchanged(self):
        approved("add_sudo_rule", user="alice", commands=[SYSTEMCTL])
        f = Path(TMP) / "agent-alice"
        before = f.read_text()
        with self.assertRaises(st.SudoersError):
            st._install(f, before + "this is not valid sudoers syntax ((\n")
        self.assertEqual(f.read_text(), before)
        self.assertEqual([p.name for p in Path(TMP).iterdir() if p.name.endswith(".tmp")], [])

    def test_whole_config_failure_rolls_back_new_and_existing(self):
        """Simulate the post-install whole-config check failing and confirm we restore state."""
        f = Path(TMP) / "agent-alice"
        orig_check, orig_flag = st._visudo_check_all, st.CHECK_WHOLE_CONFIG
        st.CHECK_WHOLE_CONFIG = True
        st._visudo_check_all = lambda: (False, "simulated failure")
        try:
            out = approved("add_sudo_rule", user="alice", commands=[SYSTEMCTL])  # new file
            self.assertIn("rolled back", out["error"])
            self.assertFalse(f.exists())
        finally:
            st._visudo_check_all, st.CHECK_WHOLE_CONFIG = orig_check, orig_flag
        approved("add_sudo_rule", user="alice", commands=[SYSTEMCTL])
        before = f.read_text()
        st._visudo_check_all, st.CHECK_WHOLE_CONFIG = (lambda: (False, "simulated failure")), True
        try:
            out = approved("add_sudo_rule", user="alice", commands=[STATUS])  # existing file
            self.assertIn("rolled back", out["error"])
        finally:
            st._visudo_check_all, st.CHECK_WHOLE_CONFIG = orig_check, orig_flag
        self.assertEqual(f.read_text(), before)


if __name__ == "__main__":
    unittest.main()
