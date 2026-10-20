"""LLM metadata quality evaluator with an empty conversation context.

Fifth pipeline stage. Loads Qwen3-8B fresh, sends only the title derived from
the video file name and the VideoProfile JSON, and asks for per-dimension
scores. No synopsis and no upstream audio/color/VLM evidence are passed.
The host applies fixed weights to produce the final quality metric.
"""

from __future__ import annotations

import gc
import json
import re
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from open_enricher import storage
from open_enricher.logging_setup import get_logger
from open_enricher.stages.evaluator_core import (
    build_report,
    failed_report,
    title_from_filename,
)
from open_enricher.stages.evaluator_core.prompt import system_prompt, user_prompt
from open_enricher.timings import TimingRecorder

logger = get_logger(__name__)


class EvaluatorStage:
    """Score a VideoProfile with a fresh LLM chat and host-side weights."""

    name = "evaluator"

    def __init__(self) -> None:
        """
        Configure device and model id for the evaluator LLM.

        The model is loaded inside ``run`` and unloaded before return so each
        experiment evaluation starts from an empty context and free VRAM.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "Qwen/Qwen3-8B"

    def _load_json(self, path) -> Any:
        """
        Load a JSON file; return None when missing or unreadable.

        Args:
            path: Filesystem path to a JSON document.

        Returns:
            Parsed object, or None on failure.
        """
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError) as err:
            logger.warning(f"Failed to parse {path.name}: {err}")
            return None

    def _clean_json_candidate(self, raw_text: str) -> str:
        """
        Isolate the JSON object between the first '{' and last '}'.

        Args:
            raw_text: Raw model completion, possibly wrapped in fences.

        Returns:
            Substring that should parse as a JSON object.
        """
        cleaned = re.sub(r"```(?:json)?", "", raw_text).strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return cleaned[start : end + 1]
        return cleaned

    def _generate_text(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        messages: List[Dict[str, str]],
        prefill: str = "{",
        max_new_tokens: int = 700,
    ) -> str:
        """
        Run a single-turn completion with assistant prefill forcing.

        Args:
            model: Loaded causal LM.
            tokenizer: Matching tokenizer.
            messages: Fresh chat (system + user only).
            prefill: Forced assistant prefix (default ``{``).
            max_new_tokens: Generation budget.

        Returns:
            Prefill plus decoded new tokens.
        """
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        forced_prompt = prompt_text + prefill
        model_inputs = tokenizer([forced_prompt], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
            )

        new_tokens = generated_ids[0][len(model_inputs.input_ids[0]) :]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return prefill + decoded

    def _free_vram(self) -> None:
        """Force garbage collection and clear the CUDA cache when available."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self, video_filename: str) -> Dict[str, Any]:
        """
        Score ``<video>_llm.json`` with an empty-context LLM and write the report.

        Args:
            video_filename: Exact file name under ``video_inputs``. The stem is
                turned into the title shown to the model.

        Returns:
            Status dict with overall score, usability flag, and output file name.
            On failure, ``status`` is ``failed`` and a zeroed report may still
            be written when possible.

        Side effects:
            Writes ``<video>_evaluator.json`` and ``<video>_evaluator_timings.json``
            under ``evaluator_results``. Loads and unloads Qwen3-8B.
        """
        logger.info(f"Starting LLM evaluator stage for {video_filename}")
        timing = TimingRecorder(self.name)
        title = title_from_filename(video_filename)
        logger.info(f"Evaluator title from filename: {title!r}")

        profile_path = storage.get_output_dir("llm") / f"{video_filename}_llm.json"
        output_dir = storage.get_output_dir(self.name)
        output_path = output_dir / f"{video_filename}_evaluator.json"
        timings_path = output_dir / f"{video_filename}_evaluator_timings.json"

        if not profile_path.exists():
            report = failed_report(
                video_filename=video_filename,
                title=title,
                profile_file=None,
                error=f"Missing LLM profile: {profile_path.name}",
            )
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, ensure_ascii=False)
            timing.save(timings_path)
            return {
                "stage": self.name,
                "status": "failed",
                "error": report["error"],
                "output_file": str(output_path.name),
                "timings_file": str(timings_path.name),
            }

        profile_data = self._load_json(profile_path)
        if profile_data is None or not isinstance(profile_data, dict):
            report = failed_report(
                video_filename=video_filename,
                title=title,
                profile_file=profile_path.name,
                error=f"Unreadable LLM profile: {profile_path.name}",
            )
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, ensure_ascii=False)
            timing.save(timings_path)
            return {
                "stage": self.name,
                "status": "failed",
                "error": report["error"],
                "output_file": str(output_path.name),
                "timings_file": str(timings_path.name),
            }

        # Fresh conversation every run: system + user only, no prior turns.
        messages = [
            {"role": "system", "content": system_prompt()},
            {
                "role": "user",
                "content": user_prompt(title=title, profile=profile_data),
            },
        ]

        quantization_config = None
        if self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        model = None
        tokenizer = None
        last_raw = ""
        try:
            logger.info(f"Loading evaluator model {self.model_id} (empty context)...")
            with timing.span("evaluator_load", model_id=self.model_id):
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id, trust_remote_code=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    quantization_config=quantization_config,
                    device_map="auto" if self.device == "cuda" else None,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
                if self.device != "cuda":
                    model = model.to(self.device)

            logger.info("Requesting dimension scores from evaluator LLM...")
            with timing.span("evaluator_generate", model_id=self.model_id):
                last_raw = self._generate_text(model, tokenizer, messages, prefill="{")
        finally:
            logger.info("Unloading evaluator model and freeing VRAM...")
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            self._free_vram()

        candidate = self._clean_json_candidate(last_raw)
        try:
            payload = json.loads(candidate)
            if not isinstance(payload, dict):
                raise ValueError("Evaluator response is not a JSON object")
            report = build_report(
                video_filename=video_filename,
                title=title,
                profile_file=profile_path.name,
                llm_payload=payload,
            )
            status = "success"
            error = None
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            raw_path = output_dir / f"{video_filename}_evaluator_raw.txt"
            raw_path.write_text(last_raw, encoding="utf-8")
            report = failed_report(
                video_filename=video_filename,
                title=title,
                profile_file=profile_path.name,
                error=f"Evaluator JSON parse failed: {exc}",
            )
            report["raw_output_file"] = raw_path.name
            status = "failed"
            error = report["error"]

        logger.info(f"Saving evaluator report to {output_path}")
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        timing.save(timings_path)

        dimension_scores = {
            name: payload["score"] for name, payload in report["dimensions"].items()
        }
        result: Dict[str, Any] = {
            "stage": self.name,
            "status": status,
            "title": title,
            "overall": report["overall"],
            "overall_normalized": report["overall_normalized"],
            "usable": report["usable"],
            "dimensions": dimension_scores,
            "output_file": str(output_path.name),
            "timings": timing.document(),
            "timings_file": str(timings_path.name),
        }
        if error:
            result["error"] = error
        return result
