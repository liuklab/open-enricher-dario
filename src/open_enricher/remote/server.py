"""FastAPI Server Entrypoint (Infrastructure Layer)."""

from contextlib import asynccontextmanager
import os
import threading
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
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
    "judger": JudgerStage()
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Pre-flight checks and database setup on boot."""
    print("[INFO] Starting worker server...")
    db.setup_database()

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