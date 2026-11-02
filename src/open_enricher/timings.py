"""Wall-clock timing helpers for pipeline stages.

Each stage constructs a ``TimingRecorder``, wraps heavy work in named
``span`` contexts, and writes a JSON timing document next to its artifacts.
Spans carry UTC ISO 8601 ``started_at`` and ``ended_at``. When CUDA is
visible, each span records the PyTorch VRAM peak. The experiment process
also samples, every five seconds, the driver-reported VRAM, the process
resident memory, and the average CPU cores in use. Each sample has a UTC
ISO 8601 ``timestamp``. The archive merges the JSON into ``timings.json``
and writes both CSVs under ``runs/local/<experiment_id>/performance/``
for external charts. One ``device.json`` in that folder records the GPUs
and the host at the start of the experiment.
"""

from __future__ import annotations

import csv
import json
import os
import resource
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List

from open_enricher.logging_setup import get_logger

logger = get_logger(__name__)

# One row per span. Fixed columns so a spreadsheet or chart tool can plot
# bars (category = span) or a histogram (values = vram_reserved_mib) without
# unnesting JSON.
PERFORMANCE_FIELDS: tuple[str, ...] = (
    "experiment",
    "video",
    "stage",
    "span",
    "window",
    "started_at",
    "ended_at",
    "seconds",
    "vram_reserved_mib",
    "vram_allocated_mib",
)

# One row every five seconds. timestamp is the wall clock of the sample.
# vram_device_mib is the driver view, ram_mib is the process resident set,
# cpu_cores is the average cores busy since the previous sample (blank on
# the first row).
VRAM_SERIES_FIELDS: tuple[str, ...] = (
    "timestamp",
    "t_seconds",
    "vram_device_mib",
    "ram_mib",
    "cpu_cores",
)

_MIB = 1024 * 1024
_DEVICE_VRAM_INTERVAL_S = 5.0


def utc_now_iso() -> str:
    """
    Return the current UTC time as an ISO 8601 string.

    Returns:
        ``datetime`` ISO format with a ``+00:00`` offset, the same shape
        written on progress events.
    """
    return datetime.now(timezone.utc).isoformat()


class TimingRecorder:
    """
    Accumulate named timing spans for one stage run.

    The wall clock starts at construction. Each ``span`` records elapsed
    seconds, UTC ISO 8601 start and end, the PyTorch VRAM peak, and
    optional metadata keys that are useful in later analysis.
    Driver-reported VRAM over time is recorded separately, once every five
    seconds, for the whole video run.
    """

    def __init__(self, stage: str) -> None:
        """
        Start recording for a pipeline stage.

        Args:
            stage: Stable stage name (audio, color, vlm, llm).
        """
        self.stage = stage
        self.entries: List[Dict[str, Any]] = []
        self._wall_start = time.perf_counter()
        self._started_at = utc_now_iso()

    @contextmanager
    def span(self, name: str, **extra: Any) -> Generator[None, None, None]:
        """
        Time a named block and append it to ``entries``.

        Args:
            name: Short label for the work unit (model load, inference, ...).
            **extra: Optional fields stored alongside the duration.

        Yields:
            Control to the caller for the timed block. Timing is always
            recorded, even when the block raises. A VRAM read failure does
            not discard the span and does not propagate.
        """
        try:
            _reset_vram_peaks()
        except Exception as exc:  # noqa: BLE001 - metrics must not stop the stage
            logger.warning("VRAM peak reset skipped: %s", exc)
        started_at = utc_now_iso()
        started = time.perf_counter()
        try:
            yield
        finally:
            ended_at = utc_now_iso()
            elapsed = time.perf_counter() - started
            row: Dict[str, Any] = {
                "name": name,
                "started_at": started_at,
                "ended_at": ended_at,
                "seconds": round(elapsed, 3),
            }
            row.update(extra)
            peaks = _safe_vram_peaks()
            if peaks:
                row.update(peaks)
            self.entries.append(row)
            if peaks:
                logger.info(
                    "TIMING %s.%s=%.3fs vram_reserved_mib=%s vram_allocated_mib=%s",
                    self.stage,
                    name,
                    elapsed,
                    peaks["vram_reserved_mib"],
                    peaks["vram_allocated_mib"],
                )
            else:
                logger.info("TIMING %s.%s=%.3fs", self.stage, name, elapsed)

    def document(self) -> Dict[str, Any]:
        """
        Build the serializable timing document for this stage run.

        Returns:
            Dict with stage name, UTC start and end, total wall seconds,
            and the span list.
        """
        return {
            "stage": self.stage,
            "started_at": self._started_at,
            "ended_at": utc_now_iso(),
            "wall_seconds": round(time.perf_counter() - self._wall_start, 3),
            "spans": list(self.entries),
        }

    def save(self, path: Path) -> Path:
        """
        Write the timing document as indented JSON.

        Args:
            path: Destination file. Parent directories are created as needed.

        Returns:
            The same ``path`` for chaining.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.document(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path


def _cuda_device_indexes() -> list[int]:
    """
    Return CUDA device indexes visible to this process.

    Returns:
        Empty list when torch is missing or CUDA is not available.
    """
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    try:
        count = int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001 - CUDA query must not stop the stage
        return []
    return list(range(count))


def _reset_vram_peaks() -> None:
    """
    Set the PyTorch VRAM peak counters to the memory held right now.

    The next ``max_memory_*`` read is then the high-water mark of the span,
    including weights already resident on the device.
    """
    indexes = _cuda_device_indexes()
    if not indexes:
        return
    import torch

    for index in indexes:
        torch.cuda.reset_peak_memory_stats(index)


def _vram_peaks_mib() -> dict[str, float] | None:
    """
    Read the PyTorch allocator peak across visible GPUs.

    Returns:
        ``vram_reserved_mib`` (caching allocator) and ``vram_allocated_mib``
        (live tensors), in mebibytes, for the device with the highest reserved
        peak. ``None`` when CUDA is not available.
    """
    indexes = _cuda_device_indexes()
    if not indexes:
        return None
    import torch

    reserved = 0
    allocated = 0
    for index in indexes:
        device_reserved = int(torch.cuda.max_memory_reserved(index))
        device_allocated = int(torch.cuda.max_memory_allocated(index))
        if device_reserved >= reserved:
            reserved = device_reserved
            allocated = device_allocated
    return {
        "vram_reserved_mib": round(reserved / _MIB, 1),
        "vram_allocated_mib": round(allocated / _MIB, 1),
    }


def _read_device_used_bytes() -> tuple[int, int] | None:
    """
    Read VRAM used and total as reported by the CUDA driver.

    Returns:
        ``(used_bytes, total_bytes)`` for the visible GPU with the most
        memory in use. ``used`` is ``total - free`` from ``mem_get_info``,
        so it includes every process on that card. ``None`` when CUDA is
        not available.
    """
    indexes = _cuda_device_indexes()
    if not indexes:
        return None
    import torch

    used = -1
    total = 0
    for index in indexes:
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        device_used = int(total_bytes) - int(free_bytes)
        if device_used >= used:
            used = device_used
            total = int(total_bytes)
    if used < 0:
        return None
    return used, total


def _resident_mib() -> float | None:
    """
    Read the resident set of this process.

    Returns:
        ``VmRSS`` in mebibytes, or ``None`` when ``/proc/self/status`` cannot
        be read.
    """
    try:
        text = Path("/proc/self/status").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            return round(int(line.split()[1]) / 1024, 1)
    return None


def _process_cpu_seconds() -> float:
    """
    Return user plus system CPU time consumed by this process.

    Returns:
        Seconds from ``resource.getrusage``. Threads of this process are
        included.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


class DeviceVramSeries:
    """
    Record driver VRAM, process RAM, and CPU every five seconds.

    The first sample is taken when the series starts. Later samples land on
    that grid (5, 10, 15, ...). Each reading is kept, with a UTC ISO 8601
    ``timestamp`` taken at the read. ``cpu_cores`` is the CPU time since the
    previous sample divided by the seconds between them. ``t_seconds`` is
    the time since ``start``, which the experiment loop aligns with one
    video run.
    """

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._rows: list[dict[str, Any]] = []
        self._thread: threading.Thread | None = None
        self._prev_cpu_seconds: float | None = None
        self._prev_t = 0.0

    def _record(self, t_seconds: float) -> None:
        reading = _read_device_used_bytes()
        vram: float | str = ""
        if reading is not None:
            used, _total = reading
            vram = round(used / _MIB, 1)
        ram = _resident_mib()
        cpu_now = _process_cpu_seconds()
        cpu_cores: float | str = ""
        if self._prev_cpu_seconds is not None and t_seconds > self._prev_t:
            elapsed = t_seconds - self._prev_t
            cpu_cores = round((cpu_now - self._prev_cpu_seconds) / elapsed, 2)
        self._prev_cpu_seconds = cpu_now
        self._prev_t = t_seconds
        row: dict[str, Any] = {
            "timestamp": utc_now_iso(),
            "t_seconds": t_seconds,
            "vram_device_mib": vram,
            "ram_mib": "" if ram is None else ram,
            "cpu_cores": cpu_cores,
        }
        with self._lock:
            self._rows.append(row)

    def start(self) -> None:
        """Take the sample at t=0 and poll every five seconds."""
        try:
            self._record(0.0)
        except Exception as exc:  # noqa: BLE001 - the poll still starts
            logger.warning("Device VRAM sample skipped: %s", exc)
        self._thread = threading.Thread(
            target=self._run,
            name="vram-device",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        step = 1
        while not self._stop.wait(_DEVICE_VRAM_INTERVAL_S):
            try:
                self._record(step * _DEVICE_VRAM_INTERVAL_S)
            except Exception as exc:  # noqa: BLE001 - keep polling until the run ends
                logger.warning("Device VRAM sample skipped: %s", exc)
            step += 1

    def stop(self) -> list[dict[str, Any]]:
        """
        Stop polling and return every sample taken.

        Returns:
            Rows of ``timestamp``, ``t_seconds``, ``vram_device_mib``,
            ``ram_mib``, and ``cpu_cores``, in time order.
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=_DEVICE_VRAM_INTERVAL_S + 1.0)
        with self._lock:
            return list(self._rows)


def start_device_vram_series() -> DeviceVramSeries | None:
    """
    Start the five-second VRAM, RAM, and CPU series when a GPU is visible.

    Returns:
        The running series, or ``None`` when CUDA is not available.
    """
    if not _cuda_device_indexes():
        return None
    series = DeviceVramSeries()
    series.start()
    return series


def _safe_vram_peaks() -> dict[str, float] | None:
    """
    Read VRAM peaks, converting any CUDA error into a skipped measurement.

    Returns:
        Peak fields, or ``None`` when the read is unavailable or fails.
    """
    try:
        return _vram_peaks_mib()
    except Exception as exc:  # noqa: BLE001 - metrics must not stop the stage
        logger.warning("VRAM peak read skipped: %s", exc)
        return None


def _load_torch() -> Any | None:
    """
    Import torch, or report that it is not installed.

    Returns:
        The torch module, or ``None`` on ``ImportError``.
    """
    try:
        import torch
    except ImportError:
        return None
    return torch


def _read_text(path: Path) -> str | None:
    """
    Read a text file, treating a missing path as no data.

    Args:
        path: File to read.

    Returns:
        File text, or ``None`` when the path cannot be read.
    """
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _optional_text(props: Any, name: str) -> str:
    """
    Read an optional device-property string.

    Args:
        props: Object returned by ``torch.cuda.get_device_properties``.
        name: Attribute name (``uuid``, ``pci_bus_id``).

    Returns:
        The attribute as text, or an empty string when it is absent.
    """
    value = getattr(props, name, None)
    if not value:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _device_row(index: int, props: Any, total_bytes: int) -> dict[str, Any]:
    """
    Build one GPU record from driver properties and ``mem_get_info``.

    Args:
        index: CUDA device index.
        props: Object returned by ``torch.cuda.get_device_properties``.
        total_bytes: Total device memory from ``mem_get_info``.

    Returns:
        JSON-serializable device fields. ``uuid`` and ``pci_bus_id`` are
        included only when this PyTorch build exposes them.
    """
    row: dict[str, Any] = {
        "index": index,
        "name": str(props.name),
        "total_memory_mib": round(int(total_bytes) / _MIB, 1),
        "compute_capability": f"{int(props.major)}.{int(props.minor)}",
        "multi_processor_count": int(props.multi_processor_count),
    }
    uuid = _optional_text(props, "uuid")
    if uuid:
        row["uuid"] = uuid
    pci_bus_id = _optional_text(props, "pci_bus_id")
    if pci_bus_id:
        row["pci_bus_id"] = pci_bus_id
    return row


def _visible_cuda_devices() -> tuple[bool, str, list[dict[str, Any]]]:
    """
    Describe every CUDA device visible to this process.

    Returns:
        ``(cuda_available, cuda_version, devices)``. ``devices`` is empty
        when CUDA is not available. Raises when a CUDA query fails.
    """
    torch = _load_torch()
    if torch is None:
        return False, "", []
    version = str(torch.version.cuda or "")
    if not torch.cuda.is_available():
        return False, version, []
    devices: list[dict[str, Any]] = []
    for index in range(int(torch.cuda.device_count())):
        props = torch.cuda.get_device_properties(index)
        _free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        devices.append(_device_row(index, props, int(total_bytes)))
    return True, version, devices


def _host_machine() -> dict[str, Any]:
    """
    Read the CPU model, logical CPU count, and total RAM.

    Returns:
        ``cpu_model``, ``cpu_count``, and ``ram_total_mib``. Model and RAM
        stay blank when ``/proc`` cannot be read. ``cpu_count`` falls back
        to ``os.cpu_count`` when ``/proc/cpuinfo`` has no processor lines.
    """
    cpu_model = ""
    cpu_count: int | str = ""
    ram_total_mib: float | str = ""
    cpuinfo = _read_text(Path("/proc/cpuinfo"))
    if cpuinfo:
        processors = 0
        for line in cpuinfo.splitlines():
            if line.startswith("model name") and not cpu_model:
                cpu_model = line.split(":", 1)[1].strip()
            elif line.startswith("processor"):
                processors += 1
        if processors:
            cpu_count = processors
    if cpu_count == "":
        fallback = os.cpu_count()
        if fallback:
            cpu_count = fallback
    meminfo = _read_text(Path("/proc/meminfo"))
    if meminfo:
        for line in meminfo.splitlines():
            if line.startswith("MemTotal:"):
                ram_total_mib = round(int(line.split()[1]) / 1024, 1)
                break
    return {
        "cpu_model": cpu_model,
        "cpu_count": cpu_count,
        "ram_total_mib": ram_total_mib,
    }


def inference_device_document() -> dict[str, Any]:
    """
    Snapshot the GPUs used for inference and the host machine.

    The experiment loop calls this once, before the first video. A CUDA
    error does not propagate: ``devices`` is empty and a warning is logged.

    Returns:
        JSON-serializable dict with ``recorded_at``, CUDA availability and
        version, one record per visible GPU, and the host CPU and RAM.
    """
    recorded_at = utc_now_iso()
    cuda_available = False
    cuda_version = ""
    devices: list[dict[str, Any]] = []
    try:
        cuda_available, cuda_version, devices = _visible_cuda_devices()
    except Exception as exc:  # noqa: BLE001 - metrics must not stop the run
        logger.warning("Inference device read skipped: %s", exc)
    return {
        "recorded_at": recorded_at,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "devices": devices,
        "host": _host_machine(),
    }


def write_performance_csv(
    path: Path,
    document: dict[str, Any],
    *,
    experiment: str,
    video: str,
) -> Path | None:
    """
    Write one flat CSV row per timing span.

    Args:
        path: Destination CSV. Parent directories are created when there is
            at least one span.
        document: Merged timing document (``{"stages": [...]}``).
        experiment: Experiment id stored on every row.
        video: Video file name stored on every row.

    Returns:
        ``path`` when a file was written, otherwise ``None``.
    """
    rows: list[dict[str, Any]] = []
    for stage_doc in document.get("stages") or []:
        if not isinstance(stage_doc, dict):
            continue
        stage = stage_doc.get("stage", "")
        for span in stage_doc.get("spans") or []:
            if not isinstance(span, dict):
                continue
            window = span.get("window", "")
            rows.append(
                {
                    "experiment": experiment,
                    "video": video,
                    "stage": stage,
                    "span": span.get("name", ""),
                    "window": "" if window is None else window,
                    "started_at": span.get("started_at", ""),
                    "ended_at": span.get("ended_at", ""),
                    "seconds": span.get("seconds", ""),
                    "vram_reserved_mib": span.get("vram_reserved_mib", ""),
                    "vram_allocated_mib": span.get("vram_allocated_mib", ""),
                }
            )
    if not rows:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=PERFORMANCE_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_vram_series_csv(path: Path, rows: list[dict[str, Any]]) -> Path | None:
    """
    Write the five-second VRAM, RAM, and CPU series.

    Args:
        path: Destination CSV. Parent directories are created when there is
            at least one sample.
        rows: Samples from ``DeviceVramSeries.stop``, in time order.

    Returns:
        ``path`` when a file was written, otherwise ``None``.
    """
    if not rows:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=VRAM_SERIES_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return path
