"""Slurm finish watcher: one post per terminal job, never twice."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "slurm_job_watcher",
    Path(__file__).resolve().parents[1] / "scripts" / "slurm_job_watcher.py",
)
assert _SPEC is not None and _SPEC.loader is not None
watcher = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = watcher
_SPEC.loader.exec_module(watcher)


def test_parse_job_ids_skips_comments_and_duplicates() -> None:
    text = "# header\n\n122347\n122347\nnot-a-job\n122349\n"
    assert watcher.parse_job_ids(text) == ["122347", "122349"]


@pytest.mark.parametrize(
    ("state", "terminal"),
    [
        ("COMPLETED", True),
        ("FAILED", True),
        ("TIMEOUT", True),
        ("CANCELLED by 1501", True),
        ("OUT_OF_MEMORY", True),
        ("NODE_FAIL", True),
        ("RUNNING", False),
        ("PENDING", False),
        ("COMPLETING", False),
        ("", False),
    ],
)
def test_terminal_states(state: str, terminal: bool) -> None:
    assert watcher.is_terminal(state) is terminal


def test_message_lists_each_job_and_carries_the_prefix() -> None:
    message = watcher.format_message(
        [
            watcher.FinishedJob("10", "open-enricher", "COMPLETED", "0:0", "00:00:05", ("/tmp/10.out",)),
            watcher.FinishedJob("11", "open-enricher", "FAILED", "1:0", "00:01:00", ()),
        ]
    )
    assert message.startswith("OPEN_ENRICHER_JOB_DONE\ncount: 2\n")
    assert "job_id=10" in message and "state=COMPLETED" in message and "exit=0:0" in message
    assert "logs=/tmp/10.out" in message
    assert "job_id=11" in message and "logs=unknown" in message
    assert "https://" not in message


def test_open_enricher_name_match() -> None:
    assert watcher.matches_open_enricher("open-enricher-watch-test")
    assert watcher.matches_open_enricher("Open-Enricher")
    assert not watcher.matches_open_enricher("smoke-bassl-himym")


def test_one_summary_then_silence(tmp_path: Path) -> None:
    track = tmp_path / "job_ids.txt"
    state = tmp_path / "state.json"
    webhook = tmp_path / "webhook"
    logs = tmp_path / "jobs_logs"
    logs.mkdir()
    (logs / "open-enricher-watch-test-10.out").write_text("ok\n", encoding="utf-8")
    track.write_text("10\n11\n12\n", encoding="utf-8")
    webhook.write_text("https://example.test/hook\n", encoding="utf-8")

    live = {
        "12": watcher.LiveJob("12", "open-enricher", "RUNNING"),
    }
    accounting = {
        "10": watcher.Accounting("10", "open-enricher-a", "COMPLETED", "0:0", "00:00:05", str(tmp_path)),
        "11": watcher.Accounting("11", "open-enricher-b", "FAILED", "1:0", "00:00:09", str(tmp_path)),
        "12": watcher.Accounting("12", "open-enricher", "RUNNING", "0:0", "00:00:01", str(tmp_path)),
    }
    sent: list[str] = []

    def poster(path: Path, text: str) -> None:
        assert path == webhook
        assert "example.test" not in text
        sent.append(text)

    def query_live(user: str) -> dict[str, watcher.LiveJob]:
        assert user == "st_mdp26"
        return dict(live)

    def query_acct(job_id: str) -> watcher.Accounting | None:
        return accounting.get(job_id)

    first = watcher.poll_once(
        track_file=track,
        state_file=state,
        webhook_file=webhook,
        user="st_mdp26",
        log_dir=logs,
        live_jobs=query_live,
        accounting=query_acct,
        poster=poster,
    )
    assert first == ["10", "11"]
    assert len(sent) == 1
    assert sent[0].count("job_id=") == 2
    assert "open-enricher-watch-test-10.out" in sent[0]

    live.clear()
    accounting["12"] = watcher.Accounting("12", "open-enricher", "TIMEOUT", "0:0", "00:02:00", str(tmp_path))
    second = watcher.poll_once(
        track_file=track,
        state_file=state,
        webhook_file=webhook,
        user="st_mdp26",
        log_dir=logs,
        live_jobs=query_live,
        accounting=query_acct,
        poster=poster,
    )
    assert second == ["12"]
    assert len(sent) == 2
    third = watcher.poll_once(
        track_file=track,
        state_file=state,
        webhook_file=webhook,
        user="st_mdp26",
        log_dir=logs,
        live_jobs=query_live,
        accounting=query_acct,
        poster=poster,
    )
    assert third == []
    assert len(sent) == 2


def test_squeue_failure_does_not_notify(tmp_path: Path) -> None:
    track = tmp_path / "job_ids.txt"
    track.write_text("10\n", encoding="utf-8")
    sent: list[str] = []

    def poster(path: Path, text: str) -> None:
        del path, text
        sent.append("sent")

    result = watcher.poll_once(
        track_file=track,
        state_file=tmp_path / "state.json",
        webhook_file=tmp_path / "webhook",
        user="st_mdp26",
        log_dir=tmp_path,
        live_jobs=lambda user: None,
        accounting=lambda job_id: watcher.Accounting("10", "n", "COMPLETED", "0:0", "00:00:01", ""),
        poster=poster,
    )
    assert result == []
    assert sent == []


def test_failed_post_is_retried(tmp_path: Path) -> None:
    track = tmp_path / "job_ids.txt"
    track.write_text("10\n", encoding="utf-8")
    calls = {"n": 0}

    def poster(path: Path, text: str) -> None:
        del path, text
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("slack post rejected with HTTP 500")

    def accounting(job_id: str) -> watcher.Accounting:
        del job_id
        return watcher.Accounting("10", "open-enricher", "COMPLETED", "0:0", "00:00:01", "")

    kwargs = dict(
        track_file=track,
        state_file=tmp_path / "state.json",
        webhook_file=tmp_path / "webhook",
        user="st_mdp26",
        log_dir=tmp_path,
        live_jobs=lambda user: {},
        accounting=accounting,
        poster=poster,
    )
    assert watcher.poll_once(**kwargs) == []
    assert watcher.poll_once(**kwargs) == ["10"]
    assert watcher.poll_once(**kwargs) == []


def test_post_slack_rejects_a_message_that_contains_a_url(tmp_path: Path) -> None:
    webhook = tmp_path / "webhook"
    webhook.write_text("https://example.test/secret-hook\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="contains a URL"):
        watcher.post_slack(webhook, "OPEN_ENRICHER_JOB_DONE\nhttps://example.test/secret-hook")
