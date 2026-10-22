# Agent 25: Sudoers Configurator

Part of the Agentic Shell project (Users, Groups & Security domain). 
Translates natural-language privilege requests into validated, least-privilege sudoers drop-in files in `/etc/sudoers.d/`.

## File Manifest

- `sudoers_tools.py`: Tool implementations, input validation, rule rendering, atomic write/rollback, approval verification, and tool schemas.
- `sudoers_agent.py`: Public interface (`SudoersAgent`), router metadata, and LLM tool loop.
- `test_sudoers_tools.py`: 28 unit tests covering input sanitization, policy enforcement, and rollback logic.
- `test_agent_loop.py`: 10 integration tests mocking the LLM loop and approval flow.
- `live_permissions_check.py`: System test validating actual `sudo -l` permissions with a temporary user.
- `_testenv.py`: Test harness configuring temp directories.

## Security Architecture

1. Structured Arguments: The LLM does not write raw sudoers syntax. It provides structured parameters to typed tools, which construct the rule.
2. Input Sanitization: Usernames, paths, and argument tokens are validated with regex allowlists to prevent newline, comma, and parameter injection.
3. Policy Enforcement: Rejects wildcards, `ALL`, unpinned subcommands for utilities like `systemctl`, and binaries that can spawn shells (`bash`, `vim`, `python`, etc.).
4. Isolated Drop-ins: Rules are written exclusively to `/etc/sudoers.d/agent-<principal>`. The main `/etc/sudoers` is never edited.
5. Atomic Gated Install: Candidate files are validated with `visudo -c -f` before install, written atomically via `os.replace` with `0440` permissions, and checked via whole-system `visudo -c` with automatic rollback on failure.
6. Cryptographic Approval: Applying changes requires human confirmation. The loop mints an HMAC token over the exact rule text, which the installer verifies before writing.
7. Concurrency and Logging: Edits are locked using `fcntl.flock`. All operations append to `/var/log/sudoers_agent.log`.

## Running Tests

Unit test suite (offline, runs in temporary directory):
```bash
sudo python3 -m unittest -v
```

Live system verification (tests real `/etc/sudoers.d/` and clean rollback):
```bash
sudo python3 live_permissions_check.py
```

## Base Shell Interface

```python
import sudoers_agent as sa

agent = sa.SudoersAgent(confirm=lambda display: shell.ask_yes_no(display))
result = agent.handle(user_prompt)

# Metadata for master router
agent_id = sa.AGENT_ID                  # 25
agent_name = sa.AGENT_NAME              # "sudoers_configurator"
description = sa.ROUTER_DESCRIPTION
```

## Scope and Limitations

- Requires Linux with `sudo`, `visudo`, and `systemd`.
- Out of scope: `Defaults`, `Cmnd_Alias`, host-specific rules, and `sudoedit`.
