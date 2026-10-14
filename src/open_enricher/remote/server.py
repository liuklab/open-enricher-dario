"""FastAPI Server Entrypoint (Infrastructure Layer)."""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# 1. Carica automaticamente TUTTI i segreti dal file iniettato dal deploy script!
# Deve essere chiamato PRIMA di inizializzare le classi.
load_dotenv("/content/.env")

from src.open_enricher.remote.storage import db
from src.open_enricher.remote.stages.interface import PipelineStage
from src.open_enricher.remote.stages.audio import AudioStage
from src.open_enricher.remote.stages.color import ColorStage
from src.open_enricher.remote.stages.vlm import VLMStage
from src.open_enricher.remote.stages.llm import LLMStage
from src.open_enricher.remote.stages.judger import JudgerStage

app = FastAPI(title="Open-Enricher Worker API")

# Retrieve the token safely from the environment (ora popolato dal file .env)
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Registry linking URL path names to class instances
STAGES: dict[str, PipelineStage] = {
    "audio": AudioStage(hf_token=HF_TOKEN),
    "color": ColorStage(max_long_scenes=120, grid_interval_sec=180),
    "vlm": VLMStage(),
    "llm": LLMStage(),
    "judger": JudgerStage()
}

class StageRequest(BaseModel):
    video_name: str

@app.on_event("startup")
def startup_event() -> None:
    """Pre-flight checks and database setup on boot."""
    print("[INFO] Starting worker server...")
    
    # Login automatico a Hugging Face se il token è presente
    if HF_TOKEN:
        try:
            from huggingface_hub import login
            login(token=HF_TOKEN, add_to_git_credential=False)
            print("[INFO] Login a Hugging Face completato via .env iniettato dal client.")
        except ImportError:
            print("[WARNING] huggingface_hub non installato, login saltato.")
    else:
        print("[WARNING] Nessun HF_TOKEN trovato nel file .env. I download potrebbero essere lenti o fallire.")
        
    db.setup_database()
    db.configure_environment()

@app.post("/stage/{stage_name}")
def execute_stage(stage_name: str, payload: StageRequest) -> dict[str, object]:
    """Triggers a specific pipeline stage."""
    if stage_name not in STAGES:
        print(f"[ERROR] Requested stage not found: {stage_name}")
        raise HTTPException(status_code=404, detail=f"Stage '{stage_name}' not registered.")
    
    stage = STAGES[stage_name]
    video_filename = payload.video_name
    
    print(f"[INFO] Received request to run '{stage_name}' on '{video_filename}'.")
    
    try:
        return stage.run(video_filename)
    except FileNotFoundError as e:
        print(f"[ERROR] File missing: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[ERROR] Stage '{stage_name}' failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal stage error: {e}")