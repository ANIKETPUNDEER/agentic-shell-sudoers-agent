"""Shared test setup: point the tool layer at a temp dir BEFORE sudoers_tools is imported."""
import os
import tempfile

TMP = os.environ.setdefault("SUDOERS_AGENT_DIR", tempfile.mkdtemp(prefix="sudoers_test_"))
os.environ.setdefault("SUDOERS_AGENT_LOG", os.path.join(TMP, "audit.log"))

import sudoers_tools as st  # noqa: E402


def approved(fn_name: str, **args):
    """Simulate the shell: compute the approval for the exact args, then run with its token."""
    ask = st.approval_text(fn_name, {**args, "apply": True})
    token = st.issue_approval(ask["text"])
    return st.dispatch(fn_name, {**args, "apply": True}, approval_token=token)
