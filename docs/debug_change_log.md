# Debug branch change log

Running log of notable commits on `debug`. Newest entries first.

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
