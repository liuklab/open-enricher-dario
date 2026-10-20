"""Audio transcription and speaker diarization using WhisperX.

First pipeline stage. Reads a video from the local data lake, produces a
timestamped dialogue list, and writes ``<video>_audio.json`` plus timings.
Later stages (VLM enrichment, LLM profiling, evaluator) consume that JSON.
"""

import gc
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch
import whisperx
from whisperx.diarize import DiarizationPipeline, assign_word_speakers

from open_enricher.storage import db
from open_enricher.timings import TimingRecorder


@dataclass
class DialogueLine:
    """One timed dialogue line after transcription and optional diarization."""

    start_timestamp: float
    end_timestamp: float
    speaker: str
    text: str


class AudioStage:
    """Transcribe, align, and optionally diarize audio from a source video."""

    name = "audio"

    def __init__(self, hf_token: str):
        """
        Configure device and WhisperX compute settings.

        Args:
            hf_token: Hugging Face token for gated pyannote diarization weights.
                When the token cannot download those weights, the stage still
                saves the transcript and marks speakers as UNKNOWN.
        """
        self.hf_token = hf_token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 16
        self.compute_type = "float16" if self.device == "cuda" else "int8"

    def _free_vram(self) -> None:
        """Force garbage collection and clear the PyTorch CUDA cache."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self, video_filename: str) -> Dict[str, Any]:
        """
        Transcribe one video and write dialogue JSON into the data lake.

        Args:
            video_filename: Exact file name under ``video_inputs``.

        Returns:
            Status dict with line count, output file name, and timings.

        Raises:
            FileNotFoundError: When the video is missing from the data lake.
        """
        video_path = db.video_inputs / video_filename
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found in data lake: {video_path}")

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
            line_text = segment.get("text", "").strip()

            if line_text:
                dialogues.append(
                    DialogueLine(
                        start_timestamp=round(segment["start"], 2),
                        end_timestamp=round(segment["end"], 2),
                        speaker=speaker,
                        text=line_text,
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
