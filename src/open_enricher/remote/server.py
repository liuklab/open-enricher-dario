"""FastAPI Server Entrypoint (Infrastructure Layer)."""

from contextlib import asynccontextmanager
import os
import sys
import threading
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

# Load secrets and cache paths before importing transformers or torch.
_env_file = os.environ.get("OPEN_ENRICHER_ENV", "/content/.env")
if os.path.isfile(_env_file):
    load_dotenv(_env_file)

from src.open_enricher.remote.storage import db

db.configure_environment()

from src.open_enricher.remote.stages.interface import PipelineStage
from src.open_enricher.remote.stages.audio import AudioStage
from src.open_enricher.remote.stages.color import ColorStage
from src.open_enricher.remote.stages.vlm import VLMStage
from src.open_enricher.remote.stages.llm import LLMStage
from src.open_enricher.remote.stages.judger import JudgerStage

HF_TOKEN = os.getenv("HF_TOKEN", "")

STAGES: dict[str, PipelineStage] = {
    "audio": AudioStage(hf_token=HF_TOKEN),
    "color": ColorStage(max_long_scenes=120, grid_interval_sec=180),
    "vlm": VLMStage(),
    "llm": LLMStage(),
    "judger": JudgerStage(),
}


class _TeeStream:
    """Mirror stdout/stderr into the experiment worker log when configured."""

    def __init__(self, primary, log_path: Path) -> None:
        self._primary = primary
        self._log_path = log_path
        self._lock = threading.Lock()

    def write(self, data: str) -> int:
        with self._lock:
            self._primary.write(data)
            self._primary.flush()
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "a", encoding="utf-8") as handle:
                handle.write(data)
        return len(data)

    def flush(self) -> None:
        self._primary.flush()

    def fileno(self):
        return self._primary.fileno()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Pre-flight checks and database setup on boot."""
    print("[INFO] Starting worker server...")
    db.setup_database()

    log_path = db.experiment_log
    sys.stdout = _TeeStream(sys.__stdout__, log_path)
    sys.stderr = _TeeStream(sys.__stderr__, log_path)
    print(f"[INFO] Worker log mirroring to {log_path}")

    if HF_TOKEN:
        try:
            from huggingface_hub import login

            login(token=HF_TOKEN, add_to_git_credential=False)
            print("[INFO] Hugging Face login completed from the injected .env file.")
        except ImportError:
            print("[WARNING] huggingface_hub is not installed; login skipped.")
    else:
        print("[WARNING] No HF_TOKEN in the environment. Model downloads may be slow or fail.")

    yield


app = FastAPI(title="Open-Enricher Worker API", lifespan=lifespan)
_stage_lock = threading.Lock()


class StageRequest(BaseModel):
    video_name: str


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "experiment_id": os.environ.get("OPEN_ENRICHER_EXPERIMENT_ID", ""),
        "experiment_prefix": os.environ.get("OPEN_ENRICHER_EXPERIMENT_PREFIX", "exp2"),
    }


@app.get("/log", response_class=PlainTextResponse)
def read_worker_log(tail: int = 20000) -> str:
    """Return the mirrored experiment worker log (last ``tail`` characters)."""
    path = db.experiment_log
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if tail > 0 and len(text) > tail:
        return text[-tail:]
    return text


@app.get("/artifact/{stage_name}/{file_name}")
def download_artifact(stage_name: str, file_name: str) -> FileResponse:
    """Download one stage artifact (JSON, raw, sidecar, timings) from the data lake."""
    if stage_name not in STAGES:
        raise HTTPException(status_code=404, detail=f"Stage '{stage_name}' not registered.")
    if "/" in file_name or "\\" in file_name or ".." in file_name:
        raise HTTPException(status_code=400, detail="Invalid file name.")
    path = db.get_output_dir(stage_name) / file_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {file_name}")
    return FileResponse(path)


@app.post("/restore/{stage_name}")
def restore_stage_artifact(stage_name: str, payload: dict) -> dict[str, object]:
    """Write a cached stage JSON into the data lake (audio/color reuse across experiments)."""
    if stage_name not in ("audio", "color"):
        raise HTTPException(status_code=400, detail="Only audio and color may be restored.")
    video_name = str(payload.get("video_name") or "").strip()
    document = payload.get("document")
    if not video_name or document is None:
        raise HTTPException(status_code=400, detail="video_name and document are required.")
    import json as _json

    out = db.get_output_dir(stage_name) / f"{video_name}_{stage_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[INFO] Restored cached {stage_name} artifact for {video_name}")
    return {"stage": stage_name, "status": "restored", "output_file": out.name}


@app.post("/stage/{stage_name}")
def execute_stage(stage_name: str, payload: StageRequest) -> dict[str, object]:
    """Triggers a specific pipeline stage."""
    if stage_name not in STAGES:
        print(f"[ERROR] Requested stage not found: {stage_name}")
        raise HTTPException(status_code=404, detail=f"Stage '{stage_name}' not registered.")

    stage = STAGES[stage_name]
    video_filename = payload.video_name

    print(f"[INFO] Received request to run '{stage_name}' on '{video_filename}'.")

    if not _stage_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A stage is already running on this GPU.")

    try:
        return stage.run(video_filename)
    except FileNotFoundError as e:
        print(f"[ERROR] File missing: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[ERROR] Stage '{stage_name}' failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal stage error: {e}")
    finally:
        _stage_lock.release()
