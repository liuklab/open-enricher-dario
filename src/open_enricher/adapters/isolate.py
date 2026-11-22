"""Run one adapter call in a child process.

A native crash in ModelScope, InsightFace or TalkNet then becomes a
fallback outcome instead of killing the pipeline.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any


def call_in_child(
    module: str,
    function: str,
    payload: dict[str, Any],
    *,
    timeout: float = 7200,
) -> dict[str, Any]:
    """
    Call ``module.function(payload)`` in a fresh interpreter.

    The child reserves its stdout pipe for the JSON response and sends
    adapter diagnostics to stderr. A crash, timeout, or unreadable response
    becomes ``status=fallback``.
    """
    code = (
        "import importlib, json, os, sys\n"
        "payload = json.loads(sys.stdin.read() or '{}')\n"
        "with os.fdopen(os.dup(1), 'w', encoding='utf-8') as response:\n"
        "    os.dup2(2, 1)\n"
        "    fn = getattr(importlib.import_module(sys.argv[1]), sys.argv[2])\n"
        "    json.dump(fn(payload), response)\n"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code, module, function],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {"status": "fallback", "reason": f"child timed out: {exc}"}
    except OSError as exc:
        return {"status": "fallback", "reason": str(exc)}
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        return {
            "status": "fallback",
            "reason": detail[-2000:] or f"child exited {completed.returncode}",
            "returncode": completed.returncode,
        }
    try:
        document = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"status": "fallback", "reason": "child returned no JSON"}
    if not isinstance(document, dict):
        return {"status": "fallback", "reason": "child returned a non-object"}
    return document


def echo_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the payload. Used to prove the child channel works."""
    return {"status": "success", "payload": payload}


def crash_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Terminate the child without a Python exception."""
    del payload
    os._exit(3)
