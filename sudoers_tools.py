"""
Sudoers Configurator Agent - tool layer (Agent #25, Users, Groups & Security).

Design principle: the LLM NEVER writes raw sudoers text. It fills in structured
fields (who, which commands, password or not). This module builds the rule
itself, validates every field against strict allowlists, and only installs the
result if `visudo -c` accepts it. A bad LLM guess can therefore at worst produce
a rejected request, never a corrupted /etc/sudoers or an injected rule.

Safety layers
  1. Input allowlists   - user/group names, absolute command paths and argument
                          tokens are regex-checked (blocks newline/`,`/`:`/`=`
                          injection that could smuggle in extra rules).
  2. Policy             - refuses ALL, wildcards, and binaries that give a root
                          shell (bash, vi, find, python ...) unless explicitly
                          overridden.
  3. Drop-in file       - writes /etc/sudoers.d/agent-<name>, never edits the
                          main /etc/sudoers.
  4. visudo -cf         - candidate file is syntax-checked BEFORE it is installed.
  5. Atomic install     - temp file (dotted name, ignored by sudo) -> chmod 0440
                          -> os.replace(); whole config re-checked afterwards and
                          rolled back on failure.
  6. Two-step apply     - apply=False (default) only previews; nothing is written.
  7. Audit log + flock  - every change is logged; concurrent edits are serialised.
"""
from __future__ import annotations

import fcntl
import grp
import hashlib
import hmac
import json
import os
import pwd
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Configuration (overridable via env vars so tests never touch the real system)
# --------------------------------------------------------------------------- #
SUDOERS_DIR = Path(os.environ.get("SUDOERS_AGENT_DIR", "/etc/sudoers.d"))
MAIN_SUDOERS = Path(os.environ.get("SUDOERS_AGENT_MAIN", "/etc/sudoers"))
AUDIT_LOG = Path(os.environ.get("SUDOERS_AGENT_LOG", "/var/log/sudoers_agent.log"))
VISUDO = os.environ.get("SUDOERS_AGENT_VISUDO", "visudo")
FILE_PREFIX = "agent-"
MANAGED_HEADER = "# Managed by Sudoers Configurator Agent - edit via the agent, not by hand."
# Run the whole-config `visudo -c` after each install (and roll back on failure).
# Defaults to on only for the real /etc/sudoers.d; tests can force it on.
CHECK_WHOLE_CONFIG = SUDOERS_DIR == Path("/etc/sudoers.d")

# --------------------------------------------------------------------------- #
# Human-approval tokens
#
# The LLM can request an apply, but it can never authorise one. A token is an
# HMAC over the EXACT action text the human was shown, keyed with a per-process
# secret the model has no access to. add/remove with apply=True verify the token
# against the rule they are about to install, so a rule that differs from the
# approved one (even by one character) is rejected.
# --------------------------------------------------------------------------- #
_APPROVAL_SECRET = os.urandom(32)


def issue_approval(action_text: str) -> str:
    """Called by the shell/agent loop ONLY after a human said yes to `action_text`."""
    return hmac.new(_APPROVAL_SECRET, action_text.encode(), hashlib.sha256).hexdigest()


def _require_approval(action_text: str, token: str | None) -> None:
    if not token or not hmac.compare_digest(issue_approval(action_text), token):
        raise SudoersError(
            "No valid human approval for this exact change. Nothing was changed. "
            f"Approval must be given for: {action_text}"
        )


def _allowlist() -> set[str]:
    """Optional hard allowlist of command basenames, e.g. SUDOERS_AGENT_ALLOWLIST=systemctl,journalctl"""
    return {x.strip() for x in os.environ.get("SUDOERS_AGENT_ALLOWLIST", "").split(",") if x.strip()}

# --------------------------------------------------------------------------- #
# Validation primitives
# --------------------------------------------------------------------------- #
USER_RE = re.compile(r"^[a-z_][a-z0-9_-]{0,31}$")
GROUP_RE = re.compile(r"^%[a-z_][a-z0-9_-]{0,31}$")
PATH_RE = re.compile(r"^/[A-Za-z0-9_./+-]+$")
ARG_RE = re.compile(r"^[A-Za-z0-9_.@/+-]+$")  # one argument token

# Programs that can spawn a root shell or read/write arbitrary files.
DANGEROUS_BINARIES = {
    "sh", "bash", "dash", "zsh", "ksh", "csh", "tcsh", "fish", "su", "sudo",
    "vi", "vim", "view", "nano", "ed", "emacs", "less", "more", "man", "pager",
    "find", "awk", "gawk", "sed", "perl", "python", "python3", "ruby", "php",
    "lua", "node", "env", "xargs", "nice", "nohup", "timeout", "watch", "time",
    "tar", "zip", "unzip", "rsync", "scp", "ssh", "ftp", "nmap", "gdb", "strace",
    "tee", "cp", "mv", "dd", "chmod", "chown", "visudo", "sudoedit",
    "mount", "umount", "docker", "podman", "git", "make", "busybox", "script",
    "screen", "tmux", "ex", "rvim", "openssl", "wget", "curl",
}
# Commands whose sub-commands are safe only when arguments are pinned.
NEEDS_ARGS = {"systemctl", "service", "journalctl", "apt", "apt-get", "dnf", "yum"}


class SudoersError(Exception):
    """Raised for any validation / policy / visudo failure."""


@dataclass
class CommandSpec:
    path: str
    args: str = ""  # space separated; "" means "no arguments allowed"

    def validate(self, allow_dangerous: bool = False) -> list[str]:
        warnings: list[str] = []
        if not PATH_RE.match(self.path):
            raise SudoersError(f"Command path must be absolute and use safe characters: {self.path!r}")
        if self.path.rstrip("/").upper().endswith("ALL"):
            raise SudoersError("'ALL' is not an acceptable command.")
        base = os.path.basename(self.path)
        allowed = _allowlist()
        if allowed and base not in allowed:
            raise SudoersError(f"'{base}' is not on this system's approved command list ({', '.join(sorted(allowed))}).")
        for tok in self.args.split():
            if tok in ("*", "?"):
                raise SudoersError("Wildcards in arguments are not allowed (they permit argument injection).")
            if not ARG_RE.match(tok):
                raise SudoersError(f"Argument token contains unsafe characters: {tok!r}")
        if self.args and self.args != self.args.strip():
            raise SudoersError("Arguments must not have leading/trailing whitespace.")
        if base in DANGEROUS_BINARIES and not allow_dangerous:
            raise SudoersError(
                f"'{base}' can be used to obtain a full root shell or read/write arbitrary files; "
                "refusing. (Set allow_dangerous=true only if the human explicitly insists.)"
            )
        if base in NEEDS_ARGS and not self.args.strip():
            raise SudoersError(
                f"'{base}' without pinned arguments would allow every sub-command. "
                f"Specify the exact arguments, e.g. 'restart apache2'."
            )
        if not Path(self.path).exists():
            warnings.append(f"{self.path} does not exist on this machine right now.")
        elif Path(self.path).is_symlink():
            warnings.append(f"{self.path} is a symlink; sudo matches the resolved binary.")
        return warnings

    def render(self) -> str:
        # sudoers: a command followed by args matches exactly those args.
        return f"{self.path} {self.args}".strip()


@dataclass
class RuleRequest:
    principal: str                 # "alice" or "%developers"
    commands: list[CommandSpec]
    nopasswd: bool = False
    runas: str = "root"
    allow_dangerous: bool = False
    warnings: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not (USER_RE.match(self.principal) or GROUP_RE.match(self.principal)):
            raise SudoersError(
                f"Invalid user/group name {self.principal!r}. Use a lowercase Unix name; prefix groups with %."
            )
        if not USER_RE.match(self.runas):
            raise SudoersError(f"Invalid run-as user {self.runas!r}.")
        if not self.commands:
            raise SudoersError("At least one command is required.")
        if len(self.commands) > 20:
            raise SudoersError("Too many commands in one rule (max 20).")
        # Existence check (warning only: the account may be created later).
        try:
            if self.principal.startswith("%"):
                grp.getgrnam(self.principal[1:])
            else:
                pwd.getpwnam(self.principal)
        except KeyError:
            self.warnings.append(f"'{self.principal}' does not exist on this system yet.")
        if self.principal in ("root",):
            raise SudoersError("root already has full access; no rule needed.")
        for c in self.commands:
            self.warnings.extend(c.validate(self.allow_dangerous))

    def render(self) -> str:
        tag = "NOPASSWD: " if self.nopasswd else "PASSWD: "
        cmds = ", ".join(c.render() for c in self.commands)
        return f"{self.principal} ALL=({self.runas}) {tag}{cmds}"


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def _run(cmd: list[str], timeout: int = 15) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, shell=False)


def _visudo_check_file(path: Path) -> tuple[bool, str]:
    """Syntax-check ONE file with `visudo -c -f`. Never touches live config."""
    if shutil.which(VISUDO) is None:
        raise SudoersError("visudo not found - refusing to modify sudoers without a validator.")
    r = _run([VISUDO, "-c", "-f", str(path)])
    return r.returncode == 0, (r.stdout + r.stderr).strip()


def _visudo_check_all() -> tuple[bool, str]:
    """Check the complete live configuration (main file + all includes)."""
    r = _run([VISUDO, "-c"])
    return r.returncode == 0, (r.stdout + r.stderr).strip()


def _filename_for(principal: str) -> str:
    # Files in sudoers.d containing '.' or ending in '~' are IGNORED by sudo,
    # so the managed name must not contain either.
    safe = principal.replace("%", "group-")
    return f"{FILE_PREFIX}{safe}"


def _audit(event: str, **data: Any) -> None:
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "uid": os.getuid(),
        "sudo_user": os.environ.get("SUDO_USER"),
        "event": event,
        **data,
    }
    try:
        AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_LOG, "a") as fh:
            fh.write(json.dumps(entry) + "\n")
    except OSError:
        pass  # auditing must never block a legitimate, validated change


class _Lock:
    """Serialise concurrent edits (two agents/terminals at once)."""

    def __enter__(self):
        SUDOERS_DIR.mkdir(parents=True, exist_ok=True)
        self.fh = open(SUDOERS_DIR / ".agent.lock", "w")
        fcntl.flock(self.fh, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.fh, fcntl.LOCK_UN)
        self.fh.close()


def _managed_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [
        ln.rstrip("\n")
        for ln in path.read_text().splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]


def _install(target: Path, new_content: str) -> dict[str, Any]:
    """
    Validate-then-install. Returns a result dict; raises SudoersError on failure
    after restoring the previous state.
    """
    previous = target.read_text() if target.exists() else None
    # Temp name contains a '.', so sudo ignores it even if we crash mid-way.
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=SUDOERS_DIR)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(new_content)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o440)
        if os.getuid() == 0:
            os.chown(tmp, 0, 0)

        ok, msg = _visudo_check_file(tmp)          # layer 4: check BEFORE install
        if not ok:
            raise SudoersError(f"visudo rejected the rule; nothing was changed.\n{msg}")

        os.replace(tmp, target)                    # layer 5: atomic swap
        if CHECK_WHOLE_CONFIG:                     # whole-system check on real config
            ok, msg = _visudo_check_all()
            if not ok:
                _rollback(target, previous)
                raise SudoersError(f"Whole-config check failed; rolled back.\n{msg}")
        return {"installed": str(target)}
    finally:
        if tmp.exists():
            tmp.unlink()


def _rollback(target: Path, previous: str | None) -> None:
    if previous is None:
        target.unlink(missing_ok=True)
    else:
        target.write_text(previous)
        os.chmod(target, 0o440)


# --------------------------------------------------------------------------- #
# TOOLS exposed to the LLM
# --------------------------------------------------------------------------- #
def detect_services(keyword: str) -> dict[str, Any]:
    """Map a vague word ('web server', 'database') to real systemd unit names."""
    if not re.match(r"^[A-Za-z0-9 _.-]{1,40}$", keyword):
        raise SudoersError("Invalid keyword.")
    aliases = {
        "web server": ["apache2", "httpd", "nginx", "lighttpd", "caddy"],
        "webserver": ["apache2", "httpd", "nginx", "lighttpd", "caddy"],
        "database": ["mysql", "mariadb", "postgresql", "mongod", "redis-server"],
        "ssh": ["ssh", "sshd"],
        "docker": ["docker"],
    }
    candidates = aliases.get(keyword.lower().strip(), [keyword.strip()])
    found: list[str] = []
    if shutil.which("systemctl"):
        try:
            r = _run(["systemctl", "list-unit-files", "--type=service", "--no-legend", "--no-pager"])
            units = {ln.split()[0].removesuffix(".service") for ln in r.stdout.splitlines() if ln.strip()}
            found = [c for c in candidates if c in units]
        except (subprocess.SubprocessError, OSError):
            pass
    return {
        "keyword": keyword,
        "installed_matches": found,
        "candidates_checked": candidates,
        "note": "If several match or none match, ask the human which service they mean."
        if len(found) != 1 else "Single match - safe to use.",
    }


def resolve_command(program: str) -> dict[str, Any]:
    """Resolve a bare program name (e.g. 'systemctl') to the absolute path sudoers needs."""
    if not re.match(r"^[A-Za-z0-9_.+-]{1,64}$", program):
        raise SudoersError("Give a bare program name like 'systemctl', not a path.")
    p = shutil.which(program)
    if not p:
        return {"program": program, "path": None, "error": "not found in PATH"}
    return {"program": program, "path": os.path.realpath(p)}


def _build_request(user: str, commands: list[dict], nopasswd: bool, run_as: str,
                   allow_dangerous: bool) -> RuleRequest:
    specs = []
    for c in commands:
        if not isinstance(c, dict) or "path" not in c:
            raise SudoersError("Each command needs a 'path' (and optional 'args').")
        specs.append(CommandSpec(path=str(c["path"]), args=str(c.get("args", "")).strip()))
    req = RuleRequest(principal=user, commands=specs, nopasswd=bool(nopasswd),
                      runas=run_as, allow_dangerous=bool(allow_dangerous))
    req.validate()
    return req


def preview_sudo_rule(user: str, commands: list[dict], nopasswd: bool = False,
                      run_as: str = "root", allow_dangerous: bool = False) -> dict[str, Any]:
    """Build + validate a rule WITHOUT writing anything. Always call this first."""
    req = _build_request(user, commands, nopasswd, run_as, allow_dangerous)
    # Dry-run the exact candidate file through visudo (no live change).
    existing = _managed_lines(SUDOERS_DIR / _filename_for(req.principal))
    rule = req.render()
    candidate = "\n".join([MANAGED_HEADER, *existing, rule]) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".sudoers", delete=False) as fh:
        fh.write(candidate)
        name = fh.name
    try:
        os.chmod(name, 0o440)
        ok, msg = _visudo_check_file(Path(name))
    finally:
        os.unlink(name)
    return {
        "rule": rule,
        "file": str(SUDOERS_DIR / _filename_for(req.principal)),
        "syntax_ok": ok,
        "visudo_output": msg,
        "warnings": req.warnings,
        "already_present": rule in existing,
        "next_step": "Show the rule to the human and ask for confirmation, then call add_sudo_rule with apply=true.",
    }


def _add_approval_text(preview: dict[str, Any]) -> str:
    return f"ADD|{preview['rule']}|{preview['file']}"


def add_sudo_rule(user: str, commands: list[dict], nopasswd: bool = False,
                  run_as: str = "root", allow_dangerous: bool = False,
                  apply: bool = False, approval_token: str | None = None) -> dict[str, Any]:
    """Install a rule. Without apply=true this is only a preview; apply needs a human-issued token
    bound to the exact rule that is about to be installed."""
    preview = preview_sudo_rule(user, commands, nopasswd, run_as, allow_dangerous)
    if not preview["syntax_ok"]:
        raise SudoersError(f"Rule failed visudo validation: {preview['visudo_output']}")
    if not apply:
        preview["applied"] = False
        return preview
    _require_approval(_add_approval_text(preview), approval_token)
    if preview["already_present"]:
        preview["applied"] = False
        preview["note"] = "Identical rule already installed; nothing to do."
        return preview

    req_principal = user
    target = SUDOERS_DIR / _filename_for(req_principal)
    with _Lock():
        lines = _managed_lines(target)
        new_content = "\n".join([MANAGED_HEADER, *lines, preview["rule"]]) + "\n"
        _install(target, new_content)
    _audit("add", principal=user, rule=preview["rule"], file=str(target))
    preview["applied"] = True
    return preview


def _remove_approval_text(user: str, removed: list[str]) -> str:
    return "REMOVE|" + user + "|" + "\n".join(removed)


def remove_sudo_rule(user: str, command_path: str | None = None, apply: bool = False,
                     approval_token: str | None = None) -> dict[str, Any]:
    """Remove one command's rule (by path) or ALL agent-managed rules for a user/group."""
    if not (USER_RE.match(user) or GROUP_RE.match(user)):
        raise SudoersError("Invalid user/group name.")
    target = SUDOERS_DIR / _filename_for(user)
    lines = _managed_lines(target)
    if not lines:
        return {"removed": [], "note": f"No agent-managed rules for {user}."}
    if command_path:
        if not PATH_RE.match(command_path):
            raise SudoersError("Invalid command path.")
        # precise match: a command token in the rule must equal the path
        keep = [ln for ln in lines if command_path not in re.split(r"[\s,]+", ln.split(")", 1)[-1])]
    else:
        keep = []
    removed = [ln for ln in lines if ln not in keep]
    result = {"would_remove": removed, "remaining": keep, "applied": False}
    if not removed:
        result["note"] = "No matching rule."
        return result
    if not apply:
        return result
    _require_approval(_remove_approval_text(user, removed), approval_token)
    with _Lock():
        if keep:
            _install(target, "\n".join([MANAGED_HEADER, *keep]) + "\n")
        else:
            previous = target.read_text()
            target.unlink()
            if CHECK_WHOLE_CONFIG:
                ok, msg = _visudo_check_all()
                if not ok:
                    target.write_text(previous)
                    os.chmod(target, 0o440)
                    raise SudoersError(f"Config invalid after removal; restored.\n{msg}")
    _audit("remove", principal=user, removed=removed)
    result["applied"] = True
    return result


def list_sudo_rules(user: str | None = None) -> dict[str, Any]:
    """List rules created by this agent (optionally for one user/group)."""
    rules: dict[str, list[str]] = {}
    if SUDOERS_DIR.exists():
        for f in sorted(SUDOERS_DIR.iterdir()):
            if f.name.startswith(FILE_PREFIX) and "." not in f.name:
                lines = _managed_lines(f)
                if user is None or any(ln.split()[0] == user for ln in lines):
                    rules[f.name] = lines
    return {"rules": rules}


def validate_sudoers() -> dict[str, Any]:
    """Run visudo -c on the live configuration."""
    ok, msg = _visudo_check_all()
    return {"ok": ok, "output": msg}


# --------------------------------------------------------------------------- #
# Function-calling schemas + dispatcher (Anthropic tool-use format)
# --------------------------------------------------------------------------- #
_CMD_ITEM = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Absolute path, e.g. /usr/bin/systemctl. Use resolve_command if unsure."},
        "args": {"type": "string", "description": "Exact allowed arguments, e.g. 'restart apache2'. Never use wildcards."},
    },
    "required": ["path"],
}
_RULE_PROPS = {
    "user": {"type": "string", "description": "Unix user (alice) or group prefixed with % (%developers)."},
    "commands": {"type": "array", "items": _CMD_ITEM, "description": "Only the commands the human asked for."},
    "nopasswd": {"type": "boolean", "description": "True only if the human said 'without a password'."},
    "run_as": {"type": "string", "description": "Target user, default root."},
}

TOOL_SCHEMAS = [
    {"name": "detect_services",
     "description": "Find real systemd service names for a vague term like 'web server'. Use before building a rule.",
     "input_schema": {"type": "object", "properties": {"keyword": {"type": "string"}}, "required": ["keyword"]}},
    {"name": "resolve_command",
     "description": "Resolve a program name (systemctl, apt) to its absolute path.",
     "input_schema": {"type": "object", "properties": {"program": {"type": "string"}}, "required": ["program"]}},
    {"name": "preview_sudo_rule",
     "description": "Build and validate a sudoers rule WITHOUT installing it. ALWAYS call this first and show the rule to the human.",
     "input_schema": {"type": "object", "properties": _RULE_PROPS, "required": ["user", "commands"]}},
    {"name": "add_sudo_rule",
     "description": "Install a validated rule via visudo -c. Set apply=true only to request installation; the shell asks the human and the install fails without their approval of this exact rule.",
     "input_schema": {"type": "object",
                      "properties": {**_RULE_PROPS, "apply": {"type": "boolean"}},
                      "required": ["user", "commands"]}},
    {"name": "remove_sudo_rule",
     "description": "Remove one command (by path) or all agent-managed rules for a user/group. apply=true only after confirmation.",
     "input_schema": {"type": "object",
                      "properties": {"user": {"type": "string"}, "command_path": {"type": "string"}, "apply": {"type": "boolean"}},
                      "required": ["user"]}},
    {"name": "list_sudo_rules",
     "description": "List rules previously created by this agent.",
     "input_schema": {"type": "object", "properties": {"user": {"type": "string"}}}},
    {"name": "validate_sudoers",
     "description": "Run visudo -c on the live configuration.",
     "input_schema": {"type": "object", "properties": {}}},
]

TOOL_FUNCTIONS = {
    "detect_services": detect_services,
    "resolve_command": resolve_command,
    "preview_sudo_rule": preview_sudo_rule,
    "add_sudo_rule": add_sudo_rule,
    "remove_sudo_rule": remove_sudo_rule,
    "list_sudo_rules": list_sudo_rules,
    "validate_sudoers": validate_sudoers,
}


_LLM_FORBIDDEN_ARGS = ("allow_dangerous", "approval_token")


def approval_text(name: str, arguments: dict[str, Any]) -> dict[str, str]:
    """
    What the human must approve for an apply request, computed from the EXACT arguments
    that will be executed (not from an earlier preview). Returns {"text", "display"}.
    """
    args = {k: v for k, v in arguments.items() if k not in ("apply", *_LLM_FORBIDDEN_ARGS)}
    if name == "add_sudo_rule":
        p = preview_sudo_rule(**args)
        if not p["syntax_ok"]:
            raise SudoersError(f"Rule failed visudo validation: {p['visudo_output']}")
        display = f"Install in {p['file']}:\n  {p['rule']}"
        if p["warnings"]:
            display += "\nWarnings: " + "; ".join(p["warnings"])
        return {"text": _add_approval_text(p), "display": display}
    if name == "remove_sudo_rule":
        r = remove_sudo_rule(**args)
        removed = r.get("would_remove", [])
        if not removed:
            raise SudoersError(r.get("note", "Nothing to remove."))
        return {"text": _remove_approval_text(args["user"], removed),
                "display": "Remove these rules:\n  " + "\n  ".join(removed)}
    raise SudoersError(f"{name} does not need approval.")


def dispatch(name: str, arguments: dict[str, Any], approval_token: str | None = None) -> dict[str, Any]:
    """
    Execute a tool call from the LLM. Errors are returned as data, not raised.
    `approval_token` is passed out-of-band by the agent loop after a human said yes;
    it can never arrive through the model-controlled `arguments`.
    """
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return {"error": f"Unknown tool {name}"}
    bad = [k for k in _LLM_FORBIDDEN_ARGS if k in arguments]
    if bad:
        return {"error": f"{', '.join(bad)} cannot be set by the agent; it is reserved for the human."}
    if approval_token is not None and name in ("add_sudo_rule", "remove_sudo_rule"):
        arguments = {**arguments, "approval_token": approval_token}
    try:
        return fn(**arguments)
    except SudoersError as e:
        return {"error": str(e)}
    except TypeError as e:
        return {"error": f"Bad arguments: {e}"}
    except PermissionError:
        return {"error": "Permission denied - the shell must run this agent as root (or via sudo)."}
    except subprocess.TimeoutExpired:
        return {"error": "Command timed out."}
