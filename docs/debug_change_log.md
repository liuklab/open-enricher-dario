# Debug branch change log

Running log of notable commits on `debug`. Newest entries first.

## 2026-11-21

### TalkNet forward is bound

- Smoke 122350 loaded `pretrain_TalkSet.model` and returned `status=fallback` with `TalkNet forward is not bound`: 1989 face tracks, 0 scores.
- The checkpoint is the TalkNet-ASD graph (visual 3D-ResNet, depthwise temporal stack, SE-ResNet audio encoder, cross-attention, `lossAV`). `src/open_enricher/adapters/talknet_asd.py` matches those names. `load_state_dict(..., strict=True)` succeeds, and a 25-frame forward returns one speaking logit per frame.
- A score is the mean speaking logit over that one-second window. `reliable` means the mean is at least 0, the same speaking cut the TalkNet demo uses. The character graph overlaps reliable windows with diarized lines and, after two hits on the same face, writes a supported `speaks` edge. `compact_account` includes that edge, so the profile can say which face character is `SPEAKER_00` instead of leaving the label unattached. A failed load still keeps the face nodes and leaves speech unresolved.
- The probe does not rescore the 1989 tracks. It cuts 8 seconds of HIMYM S01E02 from 00:00:20 (the couch scene, where the transcript already has `SPEAKER_00`), samples at most 40 frames, and forwards at most 4 windows. Boxes smaller than 40 pixels are dropped.
- Job `122566` (`smoke-talknet-forward`, 20-minute cap, 1 GPU) is pending with reason Priority. The finish watcher is tracking that id. This pass did not wait on the InsightFace CUDA probe.

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
