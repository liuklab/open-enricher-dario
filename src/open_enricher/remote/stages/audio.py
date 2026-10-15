"""Audio extraction and diarization using WhisperX (Domain Layer)."""

import gc
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch
import whisperx
from whisperx.diarize import DiarizationPipeline, assign_word_speakers

from src.open_enricher.remote.storage import db
from src.open_enricher.remote.timings import TimingRecorder


@dataclass
class DialogueLine:
    start_timestamp: float
    end_timestamp: float
    speaker: str
    text: str


class AudioStage:
    """Handles audio transcription, alignment, and diarization."""

    name = "audio"

    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 16
        self.compute_type = "float16" if self.device == "cuda" else "int8"

    def _free_vram(self) -> None:
        """Forces garbage collection and clears PyTorch CUDA cache."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self, video_filename: str) -> Dict[str, Any]:
        video_path = db.video_inputs / video_filename
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found in Data Lake: {video_path}")

        timing = TimingRecorder(self.name)
        print("[INFO] Loading WhisperX model...")
        with timing.span("whisperx_load", model_id="large-v2"):
            model = whisperx.load_model("large-v2", self.device, compute_type=self.compute_type)

        try:
            print(f"[INFO] Reading audio from: {video_path}")
            audio = whisperx.load_audio(str(video_path))

            print("[INFO] Transcribing...")
            with timing.span("whisperx_transcribe", model_id="large-v2"):
                result = model.transcribe(
                    audio, batch_size=self.batch_size, interleaved_context=True
                )
        finally:
            del model
            self._free_vram()

        print("[INFO] Performing temporal alignment...")
        with timing.span("whisperx_align_load"):
            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"], device=self.device
            )
        try:
            with timing.span("whisperx_align"):
                result = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio,
                    self.device,
                    return_char_alignments=False,
                )
        finally:
            del model_a
            self._free_vram()

        print("[INFO] Identifying speakers...")
        os.environ["HF_TOKEN"] = self.hf_token
        diarize_model = None
        try:
            with timing.span("diarization_load"):
                try:
                    diarize_model = DiarizationPipeline(token=self.hf_token, device=self.device)
                except TypeError:
                    diarize_model = DiarizationPipeline(device=self.device)
            with timing.span("diarization_infer"):
                diarize_segments = diarize_model(audio)
                result = assign_word_speakers(diarize_segments, result)
        except Exception as exc:
            # The community diarization weights are gated. The transcript is still usable.
            print(f"[WARNING] Speaker diarization skipped: {exc}")
        finally:
            if diarize_model is not None:
                del diarize_model
            self._free_vram()

        print("[INFO] Structuring data payload...")
        dialogues = []
        for segment in result["segments"]:
            speaker = segment.get("speaker", "UNKNOWN")
            testo = segment.get("text", "").strip()

            if testo:
                dialogues.append(
                    DialogueLine(
                        start_timestamp=round(segment["start"], 2),
                        end_timestamp=round(segment["end"], 2),
                        speaker=speaker,
                        text=testo,
                    )
                )

        payload = [asdict(d) for d in dialogues]
        output_dir = db.get_output_dir(self.name)
        output_path = output_dir / f"{video_filename}_audio.json"
        timings_path = output_dir / f"{video_filename}_audio_timings.json"
        timing.save(timings_path)

        print(f"[INFO] Saving results to {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        return {
            "stage": self.name,
            "status": "success",
            "lines_extracted": len(dialogues),
            "result": payload,
            "timings": timing.document(),
            "output_file": str(output_path.name),
            "timings_file": str(timings_path.name),
        }
