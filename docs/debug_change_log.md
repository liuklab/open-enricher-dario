# Debug branch change log

Running log of commits pushed to `origin/debug` for the GitHub↔cluster loop. Newest entries first.

## 2026-11-20

### next_run: InsightFace HIMYM smoke

- Added `next_run/smoke_insightface_himym.sh` (GPU sbatch) for HIMYM face reappearance probe via `test_scripts/insightface_faces.py`.

### next_run: BaSSL HIMYM smoke

- Added `next_run/smoke_bassl_himym.sh` (GPU sbatch) for `How_I_Met_Your_Mother.S01E02.mkv` via `test_scripts/bassl_scenes.py`.

### test_scripts: EVA/Q-Former encode smoke

- Added `test_scripts/eva_encode.py`: calls `encode_video`; exits 0 only on real tokens or the known unbound-forward fallback with checkpoint present.

### test_scripts: TalkNet speaker smoke

- Added `test_scripts/talknet_speakers.py`: detect face tracks then call `active_speakers`; exits 0 only when scoring succeeds or the known unbound-forward fallback is returned with checkpoint present.

### requirements: add decord for BaSSL

- Added unpinned `decord` under the BaSSL/ModelScope block so the movie-scene video preprocessor can import without pulling `modelscope[cv]` (which would break the pinned GPU stack).

### Bootstrap this log

- Added `docs/debug_change_log.md` so each smoke/setup change on `debug` gets a dated entry.
