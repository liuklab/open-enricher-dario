"""Base protocol for all pipeline stages (Interface Layer)."""

from typing import Any, Dict, Protocol

class PipelineStage(Protocol):
    """Contract that all ML analysis stages must implement."""
    
    name: str

    def run(self, video_filename: str) -> Dict[str, Any]:
        """
        Executes the stage logic.
        
        Args:
            video_filename: The exact name of the file in the Video-Inputs folder.
            
        Returns:
            A JSON-serializable dictionary with execution metadata.
        """
        ...


