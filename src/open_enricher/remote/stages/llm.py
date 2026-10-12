"""Large Language Model profiling stage for RecSys (Domain Layer)."""

import gc
import json
import re
from typing import Any, Dict, List

import torch
from pydantic import BaseModel, Field, ValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.open_enricher.remote.storage import db


# ==========================================
# 1. DEFINIZIONE DELLO SCHEMA (PYDANTIC)
# ==========================================
class RecSysMetadata(BaseModel):
    primary_genre: str = Field(..., description="Main genre, e.g. Action, Comedy, Education, Vlog")
    sub_genres: List[str] = Field(..., description="1 to 4 specific sub-genres")
    pacing: str = Field(..., description="Fast, Moderate, or Slow")
    mood: List[str] = Field(..., description="1 to 4 mood adjectives")


class SearchAndDiscovery(BaseModel):
    semantic_tags: List[str] = Field(..., description="3 to 6 search keywords/tags")
    main_topics: List[str] = Field(..., description="1 to 4 main topics discussed or shown")


class AudienceProfiling(BaseModel):
    target_age_group: str = Field(..., description="e.g. Gen Z, Millennials, Young Adults, All Ages")
    viewer_interests: List[str] = Field(..., description="1 to 4 target audience interests")


class ContentSummary(BaseModel):
    visual_style: str = Field(..., description="Concise description of aesthetic and color palette")
    narrative_hook: str = Field(..., description="One sentence describing what makes the video engaging")


class VideoProfile(BaseModel):
    """Schema JSON finale validato per il Recommender System."""
    recsys_metadata: RecSysMetadata
    search_and_discovery: SearchAndDiscovery
    audience_profiling: AudienceProfiling
    content_summary: ContentSummary


# ==========================================
# 2. LOGICA DELLO STAGE
# ==========================================
class LLMStage:
    """Fuses metadata into guaranteed structured Recommender System profiles."""

    name = "llm"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "Qwen/Qwen3-8B"

    def _gather_metadata(self, video_filename: str) -> Dict[str, Any]:
        """Collects all available JSON outputs from previous stages."""
        enriched_script = {}

        expected_files = {
            "audio": db.get_output_dir("audio") / f"{video_filename}_audio.json",
            "color": db.get_output_dir("color") / f"{video_filename}_color.json",
            "vlm": db.get_output_dir("vlm") / f"{video_filename}_vlm.json"
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
        prefill: str = "{"
    ) -> str:
        """Runs inference with assistant prefill forcing."""
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        forced_prompt = prompt_text + prefill
        model_inputs = tokenizer([forced_prompt], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=600,
                do_sample=False,
                repetition_penalty=1.1
            )

        new_tokens = generated_ids[0][len(model_inputs.input_ids[0]):]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return prefill + decoded

    def run(self, video_filename: str) -> Dict[str, Any]:
        print(f"[INFO] Starting RecSys LLM profiling for {video_filename}")

        metadata = self._gather_metadata(video_filename)
        if not metadata:
            return {
                "stage": self.name,
                "status": "failed",
                "error": "No input metadata found from previous stages."
            }

        # JSON compatto senza spazi inutili per ridurre il consumo di token in KV-cache
        script_text = json.dumps(metadata, separators=(",", ":"), ensure_ascii=False)

        example_structure = {
            "recsys_metadata": {
                "primary_genre": "string",
                "sub_genres": ["string"],
                "pacing": "Fast | Moderate | Slow",
                "mood": ["string"]
            },
            "search_and_discovery": {
                "semantic_tags": ["string"],
                "main_topics": ["string"]
            },
            "audience_profiling": {
                "target_age_group": "string",
                "viewer_interests": ["string"]
            },
            "content_summary": {
                "visual_style": "string",
                "narrative_hook": "string"
            }
        }

        system_prompt = (
            "You are a strict JSON generator for a Video Recommender System. "
            "Output ONLY a valid JSON object matching the exact structure requested. "
            "Never include explanations or markdown blocks."
        )

        user_prompt = (
            f"Analyze the following available video metadata:\n{script_text}\n\n"
            f"Populate this exact JSON structure:\n{json.dumps(example_structure, indent=2)}"
        )

        print(f"[INFO] Loading textual LLM: {self.model_id} in 4-bit...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        final_json = None
        last_raw_response = ""

        try:
            for attempt in range(1, 3):
                print(f"[INFO] Generating structured JSON profile (Attempt {attempt}/2)...")
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
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Your JSON failed schema validation with error:\n{err}\n"
                            "Fix the JSON and output ONLY the corrected valid JSON object."
                        )
                    })
        finally:
            print("[INFO] Unloading LLM and freeing VRAM...")
            del model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if final_json is None:
            raw_path = db.get_output_dir(self.name) / f"{video_filename}_llm_raw.txt"
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(last_raw_response)
            return {
                "stage": self.name,
                "status": "failed",
                "error": "LLM failed to produce valid JSON matching schema after 2 attempts.",
                "raw_output_file": str(raw_path.name)
            }

        output_path = db.get_output_dir(self.name) / f"{video_filename}_llm.json"
        print(f"[INFO] Saving structured profile to {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_json, f, indent=2, ensure_ascii=False)

        return {
            "stage": self.name,
            "status": "success",
            "output_file": str(output_path.name)
        }