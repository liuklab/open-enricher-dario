"""FastAPI Server Entrypoint (Infrastructure Layer)."""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# 1. Carica automaticamente TUTTI i segreti dal file iniettato dal deploy script!
# Deve essere chiamato PRIMA di inizializzare le classi.
load_dotenv("/content/.env")

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