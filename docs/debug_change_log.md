# Debug branch change log

Running log of commits pushed to `origin/debug` for the GitHub↔cluster loop. Newest entries first.

## 2026-11-20

### commands: re-trigger checkpoint path check

- Comment-only touch of `commands/check_model_checkpoints.sh` so it runs once under the new watcher order (saved tip was already at `f23041d`).

### watcher: commands before sbatch

- Reordered `next_run/sync_debug.sh` so changed `commands/*.sh` run on the login node right after fetch/ff-merge (and deps), before any `sbatch` of `next_run/*.sh`. Tip advances only after all sbatch calls succeed; on sbatch failure tip stays put and existing `jobs_logs/command-*-<HEAD>.log` transcripts skip re-runs. Documented in `docs/debug_batch.md`.

### commands: checkpoint path check

- Added `commands/check_model_checkpoints.sh` (login node, 15 min, no GPU) to verify `dataset/models_cache/external/{bassl,insightface/buffalo_l,talknet,lavis}` and print non-CUDA import/version lines.

### next_run: EVA/Q-Former HIMYM smoke

- Added `next_run/smoke_eva_himym.sh` (GPU sbatch) for HIMYM visual encode probe via `test_scripts/eva_encode.py`.

### next_run: TalkNet HIMYM smoke

- Added `next_run/smoke_talknet_himym.sh` (GPU sbatch) for HIMYM TalkNet adapter probe via `test_scripts/talknet_speakers.py`.

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
