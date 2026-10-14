"""Vision-Language Model processing stage orchestrator (Domain Layer)."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any, Dict

from src.open_enricher.remote.stages.vlm_core.config import build_configs
from src.open_enricher.remote.stages.vlm_core.model import (
    extract_metadata,
    load_model,
    unload_model,
)
from src.open_enricher.remote.stages.vlm_core.video import probe_duration_seconds
from src.open_enricher.remote.storage import db


class VLMStage:
    """Handles multimodal feature extraction using Qwen3-VL in 4-bit."""

    name = "vlm"

    def _parse_json_safely(self, raw_text: str) -> tuple[dict[str, Any], bool]:
        """Extracts and parses the JSON object from raw VLM output."""
        cleaned = re.sub(r"```(?:json)?", "", raw_text).strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start : end + 1]
            candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                return json.loads(candidate), True
            except json.JSONDecodeError:
                pass

        return {"raw_unparsed_vlm_output": raw_text}, False

    def run(self, video_filename: str) -> Dict[str, Any]:
        """Executes the VLM pipeline stage and persists outputs + sidecar."""
        video_path = db.video_inputs / video_filename
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found in Data Lake: {video_path}")

        print(f"[INFO] Starting VLM stage for {video_filename}")
        model_cfg, meta_cfg = build_configs()
        duration = probe_duration_seconds(video_path)
        meta_cfg = replace(
            meta_cfg,
            prompt=(
                f"{meta_cfg.prompt}\n\n"
                f"This file lasts {duration:.0f} seconds ({duration / 60:.0f} minutes). "
                "The frames you see are spaced evenly from the first seconds to the final minutes. "
                "scene_timeline must cover that full span: the last segment ends near the end of the file, "
                "not after only the opening minutes. "
                "setting is one string, not a list. "
                "notable_scenes is a list of strings, not a list of lists."
            ),
        )

        print(f"[INFO] Loading VLM: {model_cfg.model_id} (4-bit={model_cfg.load_in_4bit})...")
        model, processor = load_model(model_cfg)

        try:
            print(
                f"[INFO] Extracting VLM metadata "
                f"(video_seconds={meta_cfg.video_seconds!r}, fps={meta_cfg.fps})..."
            )
            raw_text = extract_metadata(model, processor, video_path, meta_cfg)
        finally:
            print("[INFO] Unloading VLM and freeing VRAM...")
            unload_model(model, processor)

        output_dir = db.get_output_dir(self.name)

        # 1. Salva l'output grezzo (.txt) per ispezione e per il Judger
        raw_path = output_dir / f"{video_filename}_vlm_raw.txt"
        raw_path.write_text(raw_text, encoding="utf-8")

        # 2. Salva il JSON strutturato (.json) letto dallo stage LLM successivo
        parsed_payload, is_valid_json = self._parse_json_safely(raw_text)
        json_path = output_dir / f"{video_filename}_vlm.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(parsed_payload, f, indent=2, ensure_ascii=False)

        # 3. Salva il sidecar JSON con tutti i parametri dell'esecuzione
        sidecar_path = output_dir / f"{video_filename}_vlm_sidecar.json"
        sidecar_payload = {
            "video": str(video_path),
            "model_id": model_cfg.model_id,
            "dtype": model_cfg.dtype,
            "load_in_4bit": model_cfg.load_in_4bit,
            "device_map": model_cfg.device_map,
            "attn_implementation": model_cfg.attn_implementation,
            "fps": meta_cfg.fps,
            "video_seconds": meta_cfg.video_seconds,
            "min_frames": meta_cfg.min_frames,
            "max_frames": meta_cfg.max_frames,
            "max_video_tokens": meta_cfg.max_video_tokens,
            "video_shortest_edge": meta_cfg.video_shortest_edge,
            "video_longest_edge": meta_cfg.video_longest_edge,
            "cap_pixels_per_frame": True,
            "max_new_tokens": meta_cfg.max_new_tokens,
            "do_sample": meta_cfg.do_sample,
            "temperature": meta_cfg.temperature,
            "top_p": meta_cfg.top_p,
            "top_k": meta_cfg.top_k,
            "repetition_penalty": meta_cfg.repetition_penalty,
            "prompt_path": meta_cfg.prompt_path,
            "json_valid": is_valid_json,
        }
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(sidecar_payload, f, indent=2, ensure_ascii=False)

        print(f"[INFO] VLM stage completed. Saved to {json_path} (json_valid={is_valid_json})")
        return {
            "stage": self.name,
            "status": "success",
            "json_valid": is_valid_json,
            "result": parsed_payload,
            "output_file": str(json_path.name),
            "raw_file": str(raw_path.name),
            "sidecar_file": str(sidecar_path.name),
        }