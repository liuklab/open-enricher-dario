"""Large Language Model profiling stage for recommender metadata.

Last pipeline stage. Fuses the plot summary and the accumulated VLM scene
metadata into a validated ``VideoProfile`` written as ``<video>_llm.json``.
"""

import gc
import json
import re
from typing import Any, Dict, List

import torch
from pydantic import ValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from open_enricher import storage
from open_enricher.logging_setup import get_logger
from open_enricher.stages.llm_core.prompt import (
    profile_repair_user_prompt,
    profile_system_prompt,
    profile_user_prompt,
)
from open_enricher.stages.pipeline_spec import load_pipeline_spec
from open_enricher.stages.video_profile import VideoProfile
from open_enricher.timings import TimingRecorder

logger = get_logger(__name__)


class LLMStage:
    """Fuses metadata into guaranteed structured Recommender System profiles."""

    name = "llm"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "Qwen/Qwen3-8B"

    def _gather_metadata(self, video_filename: str) -> Dict[str, Any]:
        """Collect the plot summary and the accumulated VLM scene metadata."""
        spec = load_pipeline_spec()
        enriched_script: Dict[str, Any] = {"experiment_id": spec.experiment_id}

        dialogue_path = storage.get_output_dir("dialogue") / f"{video_filename}_dialogue.json"
        vlm_path = storage.get_output_dir("vlm") / f"{video_filename}_vlm.json"
        if dialogue_path.is_file():
            dialogue = json.loads(dialogue_path.read_text(encoding="utf-8"))
            enriched_script["plot_summary"] = dialogue.get("plot_summary", "")
        else:
            logger.info("No dialogue summary found for %s", video_filename)
        if vlm_path.is_file():
            vlm = json.loads(vlm_path.read_text(encoding="utf-8"))
            enriched_script["scene_metadata"] = vlm.get("scene_timeline", [])
        else:
            logger.info("No VLM metadata found for %s", video_filename)
        return enriched_script

    def _clean_json_candidate(self, raw_text: str) -> str:
        """Isolates the JSON object between the first '{' and last '}'."""
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
        max_new_tokens: int = 600,
    ) -> str:
        """Runs inference with assistant prefill forcing."""
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

    def run(self, video_filename: str) -> Dict[str, Any]:
        """
        Fuse upstream metadata into a validated VideoProfile JSON file.

        Args:
            video_filename: Exact file name under ``video_inputs``.

        Returns:
            Status dict. On success includes the profile and output file name.
            On failure includes an error message and optional raw output path.
        """
        logger.info(f"Starting RecSys LLM profiling for {video_filename}")
        timing = TimingRecorder(self.name)

        metadata = self._gather_metadata(video_filename)
        if not metadata.get("plot_summary") and not metadata.get("scene_metadata"):
            return {
                "stage": self.name,
                "status": "failed",
                "error": "No input metadata found from previous stages.",
            }

        script_text = json.dumps(metadata, separators=(",", ":"), ensure_ascii=False)

        example_structure = {
            "recsys_metadata": {
                "primary_genre": "string",
                "sub_genres": ["string"],
                "pacing": "Fast | Moderate | Slow",
                "mood": ["string"],
            },
            "search_and_discovery": {
                "semantic_tags": ["string"],
                "main_topics": ["string"],
            },
            "audience_profiling": {
                "target_age_group": "string",
                "viewer_interests": ["string"],
            },
            "content_summary": {
                "visual_style": "string",
                "narrative_hook": "string",
            },
        }

        logger.info(f"Loading textual LLM: {self.model_id} in 4-bit...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        with timing.span("llm_load", model_id=self.model_id):
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

        messages = [
            {"role": "system", "content": profile_system_prompt()},
            {
                "role": "user",
                "content": profile_user_prompt(
                    script_text=script_text,
                    example_structure=example_structure,
                ),
            },
        ]

        final_json = None
        last_raw_response = ""

        try:
            for attempt in range(1, 3):
                logger.info(f"Generating structured JSON profile (Attempt {attempt}/2)...")
                with timing.span("llm_generate", model_id=self.model_id, attempt=attempt):
                    raw_response = self._generate_text(model, tokenizer, messages, prefill="{")
                last_raw_response = raw_response
                candidate = self._clean_json_candidate(raw_response)

                try:
                    validated_profile = VideoProfile.model_validate_json(candidate)
                    final_json = validated_profile.model_dump()
                    logger.info("Pydantic schema validation succeeded!")
                    break
                except (ValidationError, ValueError) as err:
                    logger.warning(f"Validation failed on attempt {attempt}: {err}")
                    messages.append({"role": "assistant", "content": raw_response})
                    messages.append(
                        {
                            "role": "user",
                            "content": profile_repair_user_prompt(error=err),
                        }
                    )

        finally:
            logger.info("Unloading LLM and freeing VRAM...")
            del model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        output_dir = storage.get_output_dir(self.name)
        timings_path = output_dir / f"{video_filename}_llm_timings.json"
        timing.save(timings_path)

        if final_json is None:
            raw_path = output_dir / f"{video_filename}_llm_raw.txt"
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(last_raw_response)
            return {
                "stage": self.name,
                "status": "failed",
                "error": "LLM failed to produce valid JSON matching schema after 2 attempts.",
                "raw_output_file": str(raw_path.name),
                "timings": timing.document(),
                "timings_file": str(timings_path.name),
            }

        output_path = output_dir / f"{video_filename}_llm.json"
        logger.info(f"Saving structured profile to {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_json, f, indent=2, ensure_ascii=False)

        return {
            "stage": self.name,
            "status": "success",
            "result": final_json,
            "timings": timing.document(),
            "output_file": str(output_path.name),
            "timings_file": str(timings_path.name),
        }
