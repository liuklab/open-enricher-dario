# Debug branch change log

Running log of notable commits on `debug`. Newest entries first.

## 2026-11-21

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
