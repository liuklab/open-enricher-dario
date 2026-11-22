"""Cursor evaluator bridge lifetime and configuration."""

import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from open_enricher.evaluation.providers.cursor_grok import CursorGrokProvider


def test_evaluation_owns_a_bridge_with_a_slurm_startup_timeout(monkeypatch):
    events = []
    client = object()

    @contextmanager
    def launch_bridge(*, workspace, timeout):
        assert Path(workspace).is_dir()
        assert timeout == 120
        events.append("launch")
        try:
            yield client
        finally:
            events.append("close")

    def prompt(message, options, *, client):
        assert client is bridge_client
        assert options.tools == []
        assert "Profile" in message
        events.append("prompt")
        return SimpleNamespace(status="finished", result='{"overall": 3}')

    bridge_client = client
    monkeypatch.setitem(sys.modules, "cursor_sdk", SimpleNamespace(
        Agent=SimpleNamespace(prompt=prompt),
        AgentOptions=SimpleNamespace,
        Client=SimpleNamespace(launch_bridge=launch_bridge),
        LocalAgentOptions=SimpleNamespace,
    ))
    provider = CursorGrokProvider(api_key="test-key")
    assert provider.complete("Score the metadata", "Profile") == '{"overall": 3}'
    assert events == ["launch", "prompt", "close"]
