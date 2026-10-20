"""Base protocol for all pipeline stages.

Every enricher stage exposes a stable ``name`` and a ``run`` method that
takes a video filename from the local dataset and returns a JSON-friendly
status dict. The in-process runner in ``open_enricher.runner`` calls stages
through this contract without HTTP.
"""

from typing import Any, Dict, Protocol


class PipelineStage(Protocol):
    """
    Contract that all ML analysis stages must implement.

    Implementations live in sibling modules (audio, color, vlm, llm, evaluator).
    """

    name: str

    def run(self, video_filename: str) -> Dict[str, Any]:
        """
        Execute the stage logic for one video.

        Args:
            video_filename: Exact name of the file in the video_inputs folder.

        Returns:
            A JSON-serializable dictionary with execution metadata. Successful
            runs typically include ``status="success"`` and ``output_file``.
            Failures use ``status="failed"`` and an ``error`` message.
        """
        ...
