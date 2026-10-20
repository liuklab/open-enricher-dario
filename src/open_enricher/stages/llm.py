"""Large Language Model profiling stage for recommender metadata.

Fourth pipeline stage. Fuses audio, color, and VLM JSON (respecting
``dialogue_color_routing`` and optional claim checking) into a validated
``VideoProfile`` written as ``<video>_llm.json``.
"""

import gc
import json
import re
from typing import Any, Dict, List

import torch
from pydantic import ValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from open_enricher.stages.pipeline_spec import load_pipeline_spec
from open_enricher.stages.video_profile import VideoProfile
from open_enricher.storage import db
from open_enricher.timings import TimingRecorder


class LLMStage:
    """Fuses metadata into guaranteed structured Recommender System profiles."""

    name = "llm"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "Qwen/Qwen3-8B"

    def _gather_metadata(self, video_filename: str) -> Dict[str, Any]:
        """Collects JSON outputs according to the active pipeline routing."""
        spec = load_pipeline_spec()
        enriched_script: Dict[str, Any] = {"experiment_id": spec.experiment_id}

        expected_files = {
            "audio": db.get_output_dir("audio") / f"{video_filename}_audio.json",
            "color": db.get_output_dir("color") / f"{video_filename}_color.json",
            "vlm": db.get_output_dir("vlm") / f"{video_filename}_vlm.json",
        }

        for stage_name, file_path in expected_files.items():
            if file_path.exists():
                print(f"[INFO] Found metadata for stage: {stage_name}")
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        enriched_script[stage_name] = json.load(f)
                except Exception as e:
                    print(f"[WARNING] Failed to parse {file_path.name}: {e}")
            else:
                print(f"[INFO] No metadata found for stage: {stage_name}. Skipping.")

        if spec.inject_dialogue_color_into_llm:
            # Per-scene dialogue/colors may already sit on the VLM timeline.
            # Also attach a capped prior brief when present.
            context_path = db.get_output_dir("vlm") / f"{video_filename}_vlm_context.txt"
            if context_path.is_file():
                brief = context_path.read_text(encoding="utf-8")
                enriched_script["prior_context"] = brief[:6000]
            # Keep per-window dialogue/colors from the VLM timeline; drop bulk audio/color.
            enriched_script.pop("audio", None)
            color = enriched_script.get("color")
            if isinstance(color, list):
                enriched_script["color_frame_count"] = len(color)
                enriched_script.pop("color", None)
        else:
            # Evidence already went into the VLM; LLM sees the timeline only.
            enriched_script.pop("audio", None)
            color = enriched_script.get("color")
            if isinstance(color, list):
                enriched_script["color_frame_count"] = len(color)
                enriched_script.pop("color", None)
            context_path = db.get_output_dir("vlm") / f"{video_filename}_vlm_context.txt"
            if context_path.is_file() and not spec.dialogue_summary_to_vlm:
                # Keep a short brief only when the VLM did not already consume the summary.
                brief = context_path.read_text(encoding="utf-8")
                enriched_script["prior_context"] = brief[:4000]

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

    def _claim_check(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        profile: Dict[str, Any],
        video_filename: str,
        timing: TimingRecorder,
    ) -> Dict[str, Any]:
        """Rewrite unsupported claims using timeline + pipeline dialogue summary only."""
        vlm_path = db.get_output_dir("vlm") / f"{video_filename}_vlm.json"
        summary_path = db.get_output_dir("vlm") / f"{video_filename}_dialogue_summary.txt"
        evidence: Dict[str, Any] = {"profile": profile}
        if vlm_path.is_file():
            evidence["vlm"] = json.loads(vlm_path.read_text(encoding="utf-8"))
        if summary_path.is_file():
            evidence["dialogue_summary"] = summary_path.read_text(encoding="utf-8")[:5000]

        system_prompt = (
            "You are a strict JSON editor for a Video Recommender System. "
            "Rewrite unsupported claims in the profile using only the provided evidence. "
            "Output ONLY a valid JSON object matching the profile schema."
        )
        user_prompt = (
            "Evidence (timeline and dialogue summary produced by the pipeline, not a human synopsis):\n"
            f"{json.dumps(evidence, separators=(',', ':'), ensure_ascii=False)}\n\n"
            "Return the corrected VideoProfile JSON. Keep the same keys. "
            "Remove or soften statements the evidence does not support."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        print("[INFO] Running claim-check pass on the profile...")
        with timing.span("llm_claim_check", model_id=self.model_id):
            raw = self._generate_text(model, tokenizer, messages, prefill="{")
        candidate = self._clean_json_candidate(raw)
        try:
            return VideoProfile.model_validate_json(candidate).model_dump()
        except (ValidationError, ValueError) as err:
            print(f"[WARNING] Claim-check validation failed; keeping original profile: {err}")
            return profile

    def run(self, video_filename: str) -> Dict[str, Any]:
        """
        Fuse upstream metadata into a validated VideoProfile JSON file.

        Args:
            video_filename: Exact file name under ``video_inputs``.

        Returns:
            Status dict. On success includes the profile and output file name.
            On failure includes an error message and optional raw output path.
        """
        print(f"[INFO] Starting RecSys LLM profiling for {video_filename}")
        timing = TimingRecorder(self.name)
        spec = load_pipeline_spec()

        metadata = self._gather_metadata(video_filename)
        if not metadata:
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

        system_prompt = (
            "You are a strict JSON generator for a Video Recommender System. "
            "Output ONLY a valid JSON object matching the exact structure requested. "
            "Never include explanations or markdown blocks."
        )

        user_prompt = (
            f"Analyze the following available video metadata:\n{script_text}\n\n"
            "Hard limits, which override any longer list in the metadata:\n"
            "- recsys_metadata.primary_genre is exactly one of: Action, Adventure, Animation, "
            "Comedy, Crime, Documentary, Drama, Education, Fantasy, Horror, Music, Mystery, "
            "News, Romance, Sci-Fi, Science Fiction, Sports, Thriller, Vlog, Western.\n"
            "- recsys_metadata.pacing is exactly one of: Fast, Moderate, Slow.\n"
            "- recsys_metadata.sub_genres has 1 to 4 items. recsys_metadata.mood has 1 to 4 items.\n"
            "- search_and_discovery.semantic_tags has 3 to 6 items. "
            "search_and_discovery.main_topics has 1 to 4 items.\n"
            "- audience_profiling.viewer_interests has 1 to 4 items.\n"
            "- audience_profiling.target_age_group is exactly one of: All Ages, Children, Teens, "
            "Young Adults, Gen Z, Millennials, Adults, Family. Do not join two labels with 'and'.\n"
            "- content_summary.visual_style is one sentence and must mention the dominant palette "
            "when the metadata describes dark, near-black, or night colors.\n"
            "- content_summary.narrative_hook is one sentence. Do not say a missing person was "
            "found alive unless the metadata says that without contradiction.\n"
            "- Use spoken names only when the metadata already contains them.\n\n"
            f"Populate this exact JSON structure:\n{json.dumps(example_structure, indent=2)}"
        )

        print(f"[INFO] Loading textual LLM: {self.model_id} in 4-bit...")
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
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        final_json = None
        last_raw_response = ""

        try:
            for attempt in range(1, 3):
                print(f"[INFO] Generating structured JSON profile (Attempt {attempt}/2)...")
                with timing.span("llm_generate", model_id=self.model_id, attempt=attempt):
                    raw_response = self._generate_text(model, tokenizer, messages, prefill="{")
                last_raw_response = raw_response
                candidate = self._clean_json_candidate(raw_response)

                try:
                    validated_profile = VideoProfile.model_validate_json(candidate)
                    final_json = validated_profile.model_dump()
                    print("[INFO] Pydantic schema validation succeeded!")
                    break
                except (ValidationError, ValueError) as err:
                    print(f"[WARNING] Validation failed on attempt {attempt}: {err}")
                    messages.append({"role": "assistant", "content": raw_response})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Your JSON failed schema validation with error:\n{err}\n"
                                "Fix the JSON and output ONLY the corrected valid JSON object."
                            ),
                        }
                    )

            if final_json is not None and spec.claim_check:
                final_json = self._claim_check(
                    model, tokenizer, final_json, video_filename, timing
                )
        finally:
            print("[INFO] Unloading LLM and freeing VRAM...")
            del model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        output_dir = db.get_output_dir(self.name)
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
        print(f"[INFO] Saving structured profile to {output_path}")
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
