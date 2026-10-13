"""FastAPI worker: resolve a video file name and run one pipeline stage."""

from __future__ import annotations

import traceback
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from open_enricher.config import DRIVE_CONFIG
from open_enricher.pipeline.order import STAGE_ORDER
from open_enricher.pipeline.stages import STAGES, require_registered_stages
from open_enricher.video import resolve_video

require_registered_stages()

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


class StageRequest(BaseModel):
    """Body for POST /stages/{name}. `video` is a file name, not a path."""

    video: str = Field(min_length=1)


@app.get("/stages")
def list_stages() -> dict[str, list[str]]:
    """Return the canonical stage order."""

    return {"stages": list(STAGE_ORDER)}


@app.post("/stages/{name}")
def run_stage(name: str, request: StageRequest) -> dict[str, Any]:
    """Resolve the video in the dataset root and run the named stage."""

    stage = STAGES.get(name)
    if stage is None:
        raise HTTPException(status_code=404, detail=f"Unknown stage: {name}")

    try:
        video_path = resolve_video(request.video, DRIVE_CONFIG.root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    print(f"Running stage {name} on {video_path}")
    try:
        return stage.run(video_path)
    except Exception as exc:
        print(f"Stage {name} failed:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def start_worker_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start uvicorn. Apply nest_asyncio so a notebook event loop can host it."""

    import nest_asyncio
    import uvicorn

    nest_asyncio.apply()
    print(f"Starting worker server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    start_worker_server()


if __name__ == "__main__":
    main()
