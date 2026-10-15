"""Visual keyframe extraction and dominant color clustering (Domain Layer)."""

import json
import shutil
from typing import Any, Dict

import cv2
import numpy as np
from PIL import Image
from scenedetect import ContentDetector, detect
from sklearn.cluster import KMeans

from src.open_enricher.remote.storage import db
from src.open_enricher.remote.timings import TimingRecorder


class ColorStage:
    """Handles scene cut detection and k-means color extraction."""

    name = "color"

    def __init__(self, max_long_scenes: int = 120, grid_interval_sec: int = 180):
        self.max_long_scenes = max_long_scenes
        self.grid_interval_sec = grid_interval_sec

    def _get_dominant_colors(self, frame_rgb: np.ndarray, k: int = 3) -> list[str]:
        """Executes K-Means clustering to find the color palette."""
        small_image = cv2.resize(frame_rgb, (150, 150))
        pixels = small_image.reshape((-1, 3))

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_

        return ["#{:02x}{:02x}{:02x}".format(int(c[0]), int(c[1]), int(c[2])) for c in colors]

    def run(self, video_filename: str) -> Dict[str, Any]:
        video_path = db.video_inputs / video_filename
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found in Data Lake: {video_path}")

        timing = TimingRecorder(self.name)
        print("[INFO] Detecting scene cuts (Hybrid Sampling)...")
        with timing.span("scene_detect"):
            scene_list = detect(str(video_path), ContentDetector(threshold=27.0))

        cap = cv2.VideoCapture(str(video_path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            target_frames_set = set()

            if len(scene_list) > self.max_long_scenes:
                longest_scenes = sorted(
                    scene_list,
                    key=lambda x: x[1].get_frames() - x[0].get_frames(),
                    reverse=True
                )[:self.max_long_scenes]
            else:
                longest_scenes = scene_list

            for scene in longest_scenes:
                start = scene[0].get_frames()
                end = scene[1].get_frames()
                middle = start + ((end - start) // 2)
                target_frames_set.add(middle)

            grid_interval_frames = max(1, int(self.grid_interval_sec * fps))
            for f in range(grid_interval_frames, total_frames, grid_interval_frames):
                target_frames_set.add(f)

            # Fallback for short or static videos where no scene cut was detected.
            if not target_frames_set and total_frames > 0:
                target_frames_set.add(total_frames // 2)

            sorted_frames = sorted(target_frames_set)

            min_distance_frames = int(10 * fps)
            final_frames = []
            for f in sorted_frames:
                if not final_frames or (f - final_frames[-1]) > min_distance_frames:
                    final_frames.append(f)

            print(f"[INFO] Sampling complete. Extracting {len(final_frames)} semantic frames.")

            output_dir = db.get_output_dir(self.name)
            frames_dir = output_dir / f"{video_filename}_frames"

            if frames_dir.exists():
                shutil.rmtree(frames_dir)
            frames_dir.mkdir(parents=True)

            results = []
            with timing.span("color_extract", frames=len(final_frames)):
                for i, f_num in enumerate(final_frames):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f_num)
                    success, frame = cap.read()

                    if success:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        timestamp = round(f_num / fps, 2)
                        palette = self._get_dominant_colors(frame_rgb, k=3)

                        frame_name = f"scene_{i+1}_t{timestamp}.jpg"
                        img = Image.fromarray(frame_rgb)
                        img.save(frames_dir / frame_name)

                        results.append({
                            "scene_id": i + 1,
                            "timestamp": timestamp,
                            "frame_file": frame_name,
                            "color_palette": palette
                        })
        finally:
            cap.release()

        json_path = output_dir / f"{video_filename}_color.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        timings_path = output_dir / f"{video_filename}_color_timings.json"
        timing.save(timings_path)

        return {
            "stage": self.name,
            "status": "success",
            "keyframes_extracted": len(results),
            "result": results,
            "timings": timing.document(),
            "output_json": str(json_path.name),
            "timings_file": str(timings_path.name),
            "frames_folder": str(frames_dir.name)
        }