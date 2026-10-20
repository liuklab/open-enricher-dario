"""Open-Enricher: local video understanding pipeline for recommender metadata.

Stages run in process on a GPU host:
audio (WhisperX) -> color (scene colors) -> vlm (Qwen3-VL) -> llm (VideoProfile)
-> judger (empty-context LLM scores).

Experiment variants under ``open_enricher.experiments.variants`` are executed
in sequence by ``python -m open_enricher``.
"""

__version__ = "0.2.0"
