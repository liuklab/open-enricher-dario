"""VRAM peaks on timing spans and the flat CSV written for external charts."""

from __future__ import annotations

from pathlib import Path

import pytest

from open_enricher.timings import (
    DeviceVramSeries,
    TimingRecorder,
    write_performance_csv,
    write_vram_series_csv,
)


def test_span_records_pytorch_vram_peak(monkeypatch: pytest.MonkeyPatch) -> None:
    """A span stores the reserved and allocated peaks next to its duration."""
    monkeypatch.setattr("open_enricher.timings._reset_vram_peaks", lambda: None)
    monkeypatch.setattr(
        "open_enricher.timings._vram_peaks_mib",
        lambda: {"vram_reserved_mib": 6144.0, "vram_allocated_mib": 5900.5},
    )

    recorder = TimingRecorder("vlm")
    with recorder.span("vlm_generate", window=2, model_id="Qwen/Qwen3-VL-8B-Instruct"):
        pass

    row = recorder.entries[0]
    assert row["name"] == "vlm_generate"
    assert row["window"] == 2
    assert row["vram_reserved_mib"] == 6144.0
    assert row["vram_allocated_mib"] == 5900.5
    assert "seconds" in row


def test_span_omits_vram_when_cuda_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing CUDA leaves the span as a plain timing row."""
    monkeypatch.setattr("open_enricher.timings._reset_vram_peaks", lambda: None)
    monkeypatch.setattr("open_enricher.timings._vram_peaks_mib", lambda: None)

    recorder = TimingRecorder("audio")
    with recorder.span("whisperx_transcribe"):
        pass

    assert "vram_reserved_mib" not in recorder.entries[0]
    assert "vram_allocated_mib" not in recorder.entries[0]


def test_span_keeps_the_row_when_the_vram_read_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CUDA error while reading the peak does not drop the span or escape."""

    def _fail() -> dict[str, float]:
        raise RuntimeError("cuda busy")

    monkeypatch.setattr("open_enricher.timings._reset_vram_peaks", lambda: None)
    monkeypatch.setattr("open_enricher.timings._vram_peaks_mib", _fail)

    recorder = TimingRecorder("llm")
    with pytest.raises(ValueError):
        with recorder.span("llm_generate"):
            raise ValueError("generate failed")

    assert recorder.entries[0]["name"] == "llm_generate"
    assert "vram_reserved_mib" not in recorder.entries[0]


def test_write_performance_csv_is_one_row_per_span(tmp_path: Path) -> None:
    """The chart file is a flat table: identity columns plus duration and VRAM."""
    document = {
        "stages": [
            {
                "stage": "vlm",
                "wall_seconds": 10.0,
                "spans": [
                    {
                        "name": "vlm_load",
                        "seconds": 4.2,
                        "vram_reserved_mib": 6100.0,
                        "vram_allocated_mib": 5800.0,
                        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
                    },
                    {
                        "name": "vlm_generate",
                        "seconds": 5.5,
                        "window": 1,
                        "vram_reserved_mib": 7200.5,
                        "vram_allocated_mib": 7000.0,
                    },
                ],
            },
            {
                "stage": "audio",
                "spans": [{"name": "whisperx_transcribe", "seconds": 3.0}],
            },
        ]
    }
    path = tmp_path / "performance" / "Suits.S01E02.mkv.csv"

    written = write_performance_csv(
        path,
        document,
        experiment="06-memory-2",
        video="Suits.S01E02.mkv",
    )

    assert written == path
    text = path.read_text(encoding="utf-8")
    assert text == (
        "experiment,video,stage,span,window,seconds,vram_reserved_mib,vram_allocated_mib\n"
        "06-memory-2,Suits.S01E02.mkv,vlm,vlm_load,,4.2,6100.0,5800.0\n"
        "06-memory-2,Suits.S01E02.mkv,vlm,vlm_generate,1,5.5,7200.5,7000.0\n"
        "06-memory-2,Suits.S01E02.mkv,audio,whisperx_transcribe,,3.0,,\n"
    )


def test_vram_series_keeps_every_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each sample keeps VRAM, RAM, and the cores busy since the previous row."""
    samples = [
        (100 * 1024 * 1024, 16000 * 1024 * 1024),
        (350 * 1024 * 1024, 16000 * 1024 * 1024),
        (200 * 1024 * 1024, 16000 * 1024 * 1024),
    ]
    cursor = {"index": 0}

    def _read() -> tuple[int, int]:
        item = samples[min(cursor["index"], len(samples) - 1)]
        cursor["index"] += 1
        return item

    cpu_at = [0.0, 10.0, 12.5]
    cpu_cursor = {"index": 0}

    def _cpu() -> float:
        value = cpu_at[min(cpu_cursor["index"], len(cpu_at) - 1)]
        cpu_cursor["index"] += 1
        return value

    monkeypatch.setattr("open_enricher.timings._read_device_used_bytes", _read)
    monkeypatch.setattr("open_enricher.timings._resident_mib", lambda: 2048.0)
    monkeypatch.setattr("open_enricher.timings._process_cpu_seconds", _cpu)
    series = DeviceVramSeries()
    series._record(0.0)
    series._record(5.0)
    series._record(10.0)

    assert series.stop() == [
        {"t_seconds": 0.0, "vram_device_mib": 100.0, "ram_mib": 2048.0, "cpu_cores": ""},
        {"t_seconds": 5.0, "vram_device_mib": 350.0, "ram_mib": 2048.0, "cpu_cores": 2.0},
        {"t_seconds": 10.0, "vram_device_mib": 200.0, "ram_mib": 2048.0, "cpu_cores": 0.5},
    ]


def test_write_vram_series_csv(tmp_path: Path) -> None:
    """The trend file is two columns: time on the grid and driver VRAM."""
    path = tmp_path / "performance" / "Suits.S01E02.mkv_vram.csv"
    written = write_vram_series_csv(
        path,
        [
            {
                "t_seconds": 0.0,
                "vram_device_mib": 512.0,
                "ram_mib": 1024.0,
                "cpu_cores": "",
            },
            {
                "t_seconds": 5.0,
                "vram_device_mib": 6144.5,
                "ram_mib": 2048.5,
                "cpu_cores": 3.5,
            },
        ],
    )

    assert written == path
    assert path.read_text(encoding="utf-8") == (
        "t_seconds,vram_device_mib,ram_mib,cpu_cores\n"
        "0.0,512.0,1024.0,\n"
        "5.0,6144.5,2048.5,3.5\n"
    )


def test_write_performance_csv_skips_empty_documents(tmp_path: Path) -> None:
    """No spans means no file, so an empty run does not leave a header-only table."""
    path = tmp_path / "performance" / "empty.csv"
    assert write_performance_csv(path, {"stages": []}, experiment="x", video="y") is None
    assert not path.exists()
