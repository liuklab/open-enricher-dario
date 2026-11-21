# Debug branch change log

Running log of notable commits on `debug`. Newest entries first.

## 2026-11-21

### Mid-episode InsightFace smokes for the other films

- `test_scripts/smoke_insightface_mid30.sh` forwards extra arguments to the existing 30s sampler. The same step is used for HIMYM, Silicon Valley, and The Kingdom. Each film is its own Slurm job, so one failure does not cancel the others. Logs stay under the main checkout `jobs_logs/`.

### Small InsightFace faces are dropped

- Relative size is the larger box side divided by the matching frame side. On the HIMYM middle 30s (144 frames, 846 faces, 1920x1080) the 50th percentile is 0.13050925925925927 (min 0.048796, max 0.357500, 423 faces below).
- That percentile is the fixed constant `MIN_FACE_SIDE_RATIO` in `src/open_enricher/face_size.py`. Runtime compares each face to the constant and does not recompute a percentile per video.
- Faces below the constant are removed before identity clustering and before JSON, crops, and annotated frames. `time` and `seconds` stay on the faces that remain. No new Slurm job.

### Face times are instants

- One face on one frame is no longer written as a start/end interval. The fields are `time` (`HH:MM:SS`) and `seconds`. `end` / `end_seconds` are added only when a later frame is a different instant.
- A fraction that only copies the integer seconds, such as 178.178 from stuffing that count into `HH:MM:SS.mmm` and parsing it back, collapses to the whole second. A real frame offset (`frame index / fps`, such as 648.038) stays next to the clock. Whole seconds stay whole.
- Rewrote the stored HIMYM face times in `test_scripts/output/How_I_Met_Your_Mother.S01E02.mkv.faces.json`. No new Slurm job.

### Middle-segment InsightFace smoke

- Added `test_scripts/insightface_mid30.py` and `test_scripts/smoke_insightface_mid30.sh`. They cut 30 seconds from the middle of `How_I_Met_Your_Mother.S01E02.mkv` and score it with InsightFace at the character-graph rate (`budgets.face_sample_fps` 5, same frame step as `detect_tracks`). Raw frames, box annotations, and face crops go under `test_scripts/output/himym-s01e02-mid30/` in this worktree, separate from job 122559.
- Submitted job `122710` (`smoke-insightface-mid30`). The cut starts at 648.038038s, the file is 30.030030s (720 frames at 23.976 fps), and the sampler keeps 144 frames at a 5-frame step (0.208542s). Slurm estimated a 2026-11-21 10:46 start on barbaraliskov, when that GPU is free. The finish watcher is tracking the id.

### InsightFace smoke was on the CPU wheel

- Job 122290 (12:40, `barbaraliskov`) bound `CUDAExecutionProvider` and finished. Job 122349 (19:49, same node) scanned the episode for 30 minutes and exited 1 with only `CPUExecutionProvider`.
- At 14:05 the CPU package `onnxruntime==1.32.0` was installed on top of `onnxruntime-gpu==1.26.0`. Both own the `onnxruntime` import, and 1.32 needs CUDA 13, so the CUDA 12 libraries in this venv never loaded. The GPU dist-info was left behind.
- Removed the CPU package and reinstalled `onnxruntime-gpu==1.26.0` with `--no-deps`. `requirements.txt` already pins that wheel and already says not to install the CPU distribution.
- `test_scripts/insightface_faces.py` now prints the onnxruntime version and provider list and exits before reading the film when `CUDAExecutionProvider` is missing. `--max-frames` caps a probe. `MAX_FRAMES` on `test_scripts/smoke_insightface_himym.sh` forwards that cap. The pipeline call still reads the whole file.
- After the reinstall the login interpreter reports onnxruntime 1.26.0 with `CUDAExecutionProvider` present. The CPU distribution is gone.
- Submitted job `122559` (`insightface-cuda-probe`, `MAX_FRAMES=20`) to read a few HIMYM frames on a GPU. It is pending for Resources; `squeue --start` estimated 2026-11-21 10:38. The finish watcher is tracking that id. No second probe was submitted on top of it.

### Slurm finish watcher

- Added `scripts/slurm_job_watcher.py`. It polls every 30 seconds. It does not tail logs and it does not submit a notify job.
- Track ids by appending one job id per line to `var/slurm_watcher/job_ids.txt` (`var/` is gitignored). The watcher also follows this user's queued jobs whose name contains `open-enricher`, and remembers those ids until they finish.
- When a tracked id leaves `squeue`, `sacct` supplies State, ExitCode and Elapsed. Any terminal state (COMPLETED, FAILED, TIMEOUT, CANCELLED, OUT_OF_MEMORY, and the other terminal Slurm states) is reported. Several jobs that finish in one poll share a single Slack message prefixed `OPEN_ENRICHER_JOB_DONE`.
- `var/slurm_watcher/state.json` records ids already sent. A failed post is retried on the next poll. The webhook is read from `~/.config/open-enricher/slack_webhook` and is never written to the log or the message.
- Start it on the login node and leave it running:

```bash
cd /SLURM_shared/st_mdp26/open-enricher
tmux new-session -d -s oe-slurm-watcher -- .venv/bin/python -u scripts/slurm_job_watcher.py
```

- Log: `var/slurm_watcher/watcher.log`. A second start exits because the first process holds `var/slurm_watcher/watcher.lock`.

## 2026-11-20

### Remove GitHub↔cluster sync scaffolding

- Agent control is now via the Cursor self-hosted worker on the cluster; do not use GitHub push→sbatch as an agent bridge.
- Deleted `commands/`, `next_run/sync_debug.sh`, `scripts/push_debug_logs.py`, `docs/debug_batch.md`, `docs/debug_sync.md`, and the untracked `.github/workflows/deploy-debug.yml` bridge workflow.
- Kept useful SLURM helpers: moved `next_run/smoke_*.sh` → `test_scripts/`, `next_run/run_experiments.sh` → repo root next to `run_job.sh` / `run_job_09_12.sh`.
- Removed the `!jobs_logs/command-*.log` gitignore exception and stopped recommending cron for `sync_debug`.
- Left experiment pipeline code, variants 09–12, `test_scripts/` smokes, and `jobs_logs/` / `runs/` outputs in place.

### test_scripts: EVA/Q-Former encode smoke

- Added `test_scripts/eva_encode.py`: calls `encode_video`; exits 0 only on real tokens or the known unbound-forward fallback with checkpoint present.

### test_scripts: TalkNet speaker smoke

- Added `test_scripts/talknet_speakers.py`: detect face tracks then call `active_speakers`; exits 0 only when scoring succeeds or the known unbound-forward fallback is returned with checkpoint present.

### requirements: add decord for BaSSL

- Added unpinned `decord` under the BaSSL/ModelScope block so the movie-scene video preprocessor can import without pulling `modelscope[cv]` (which would break the pinned GPU stack).

### Bootstrap this log

- Added `docs/debug_change_log.md` so each smoke/setup change on `debug` gets a dated entry.
