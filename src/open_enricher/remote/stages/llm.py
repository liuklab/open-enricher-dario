"""Large Language Model profiling stage for RecSys (Domain Layer)."""

import gc
import json
from typing import Any, Dict, List

import torch
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Importiamo la libreria per forzare matematicamente l'output
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

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

    def run(self, video_filename: str) -> Dict[str, Any]:
        print(f"[INFO] Starting Strictly Constrained LLM profiling for {video_filename}")
        
        # Ingestion
        metadata = self._gather_metadata(video_filename)
        if not metadata:
            return {
                "stage": self.name,
                "status": "failed",
                "error": "No input metadata found from previous stages."
            }

        script_text = json.dumps(metadata, indent=2, ensure_ascii=False)
        
        # Prompt Engineering (Ora molto più semplice, perché la struttura è garantita)
        system_prompt = (
            "You are an analytical AI for a Video Recommender System. "
            "Analyze the provided timestamped video data and extract the requested metadata."
        )
        
        user_prompt = f"INPUT DATA:\n{script_text}\n\nExtract and summarize the metadata based on the input."

        # Model Loading
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
            trust_remote_code=True
        )

        print("[INFO] Preparing Constrained Generation Enforcer...")
        # Creiamo il parser basato sullo schema Pydantic
        parser = JsonSchemaParser(VideoProfile.model_json_schema())
        # Costruiamo il filtro che impedirà all'LLM di generare token errati
        prefix_function = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        text_input = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text_input], return_tensors="pt").to(self.device)

        print("[INFO] Generating structured JSON profile (Zero-Hallucination Mode)...")
        # Generazione con prefisso forzato
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=1500,
            # La magia avviene qui: questo forza ogni singolo token generato a rispettare lo schema JSON
            prefix_allowed_tokens_fn=prefix_function, 
            # Rimuoviamo il sampling per massimizzare il determinismo e lasciare fare al parser
            do_sample=False 
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # VRAM Cleanup
        print("[INFO] Unloading LLM and freeing VRAM...")
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        # Parsing (Ora è garantito al 100% che sia un JSON valido)
        print("[INFO] Parsing Guaranteed JSON output...")
        try:
            final_json = json.loads(response)
        except json.JSONDecodeError as e:
            print(f"[CRITICAL ERROR] The enforcer failed (this should be mathematically impossible): {e}")
            return {"stage": self.name, "status": "failed"}

        output_path = db.get_output_dir(self.name) / f"{video_filename}_llm.json"
        print(f"[INFO] Saving structured profile to {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_json, f, indent=2, ensure_ascii=False)

        return {
            "stage": self.name,
            "status": "success",
            "output_file": str(output_path.name)
        }