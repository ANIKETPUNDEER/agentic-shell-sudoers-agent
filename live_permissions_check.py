"""
LIVE effective-permission check (review item 3). `visudo -c` proves syntax only; this asks sudo
itself what a user may run, using the REAL /etc/sudoers.d.

    sudo python3 live_permissions_check.py

Creates a throwaway user, installs the example rule through the agent's own tool path (with
the real whole-config visudo check and rollback ON), asserts the allowed command is permitted
and unrelated privileged commands are denied, then revokes and asserts access is gone.
Cleans up the user and files even on failure. Needs root and a sudo that includes /etc/sudoers.d.
Not part of `unittest` discovery because it changes the real system config (briefly).
"""
import os
import subprocess
import sys
import tempfile

if os.geteuid() != 0:
    sys.exit("Run as root: sudo python3 live_permissions_check.py")

os.environ["SUDOERS_AGENT_DIR"] = "/etc/sudoers.d"
os.environ["SUDOERS_AGENT_LOG"] = os.path.join(tempfile.mkdtemp(), "audit.log")

import sudoers_tools as st  # noqa: E402

USER = "sudoagenttest"
SYSTEMCTL = "/usr/bin/systemctl"
ALLOWED = [SYSTEMCTL, "restart", "apache2"]
DENIED = {
    "systemctl stop apache2 (same binary, other verb)": [SYSTEMCTL, "stop", "apache2"],
    "systemctl restart ssh (same verb, other service)": [SYSTEMCTL, "restart", "ssh"],
    "systemctl restart apache2 extra-arg": [SYSTEMCTL, "restart", "apache2", "ssh"],
    "/bin/bash": ["/bin/bash"],
    "/usr/bin/id": ["/usr/bin/id"],
    "cat /etc/shadow": ["/usr/bin/cat", "/etc/shadow"],
}
failures = []


def can_run(cmd) -> bool:
    """Ask sudo whether USER may run cmd NON-interactively (-n), without running it (-l)."""
    r = subprocess.run(["sudo", "-n", "-l", "-U", USER, *cmd], capture_output=True, text=True)
    return r.returncode == 0  # sudo -l <cmd> exits 0 only if that exact command is permitted


def check(label, cond):
    print(("PASS  " if cond else "FAIL  ") + label)
    if not cond:
        failures.append(label)


def approved(fn, **args):
    ask = st.approval_text(fn, {**args, "apply": True})
    return st.dispatch(fn, {**args, "apply": True}, approval_token=st.issue_approval(ask["text"]))


try:
    subprocess.run(["useradd", "-M", "-s", "/usr/sbin/nologin", USER], check=True)
    check("baseline: user has no sudo rights", not can_run(ALLOWED))

    out = approved("add_sudo_rule", user=USER, nopasswd=True,
                   commands=[{"path": SYSTEMCTL, "args": "restart apache2"}])
    check(f"rule installed ({out.get('rule', out.get('error'))})", out.get("applied") is True)
    check("whole config still valid (visudo -c)", st.validate_sudoers()["ok"])

    check("ALLOWED: systemctl restart apache2", can_run(ALLOWED))
    for label, cmd in DENIED.items():
        check(f"DENIED:  {label}", not can_run(cmd))

    out = approved("remove_sudo_rule", user=USER)
    check("revocation applied", out.get("applied") is True)
    check("after revoke: previously allowed command now denied", not can_run(ALLOWED))
    check("whole config valid after revoke", st.validate_sudoers()["ok"])
finally:
    subprocess.run(["userdel", USER], capture_output=True)
    for name in os.listdir("/etc/sudoers.d"):
        if name.startswith("agent-" + USER):
            os.unlink(os.path.join("/etc/sudoers.d", name))

print("\nALL LIVE CHECKS PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
