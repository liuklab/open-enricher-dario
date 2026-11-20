# Planned runs on adalovelace

New runs are shell scripts in `next_run/`. An agent working through GitHub commits those scripts on `debug`. This machine fetches that branch and submits each new or changed script with `sbatch`. A push that does not add or modify one of those scripts does not start a job.

The checkout that moves is `/SLURM_shared/st_mdp26/open-enricher`. `dataset/`, `runs/`, `jobs_logs/`, `.env`, and `.venv` stay on disk. The cluster address is private, so the fetch runs here, not on a GitHub-hosted runner.

## What the other agent commits

Put one SLURM batch script directly in `next_run/`, for example `next_run/09-12.sh`. The file is the run: `#SBATCH` headers, then the `python -m open_enricher` command. Do not describe the plan only in a markdown file or a commit message. This side does not read those.

`next_run/sync_debug.sh` is the watcher. It is not a run. Do not put a job there, and do not give a run that name.

A script already on `debug` is not submitted again until a later commit changes it. Deleting a script does not cancel a job that was already queued. Editing a script and pushing it submits that script again.

`next_run/run_experiments.sh` is one planned run already in the tree. It asks for one GPU, 8 CPUs, 32 GB, and 72 hours on partition `all-nodes`, account `studenti`, and runs every variant on `How_I_Met_Your_Mother.S01E02.mkv`, `Silicon_Valley.S06E01.mkv`, and `The_Kingdom.S01E02.mkv`. A new plan is a new file next to it, with its own headers and command. `Suits.S01E02.mkv` is not in that existing script.

Submit a script by hand with:

```bash
sbatch /SLURM_shared/st_mdp26/open-enricher/next_run/run_experiments.sh
```

## What this machine does

Cron on the login node, user `st_mdp26`, runs the watcher once a minute:

```text
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
* * * * * /SLURM_shared/st_mdp26/open-enricher/next_run/sync_debug.sh >> /SLURM_shared/st_mdp26/actions-runner/sync-debug.log 2>&1
```

`next_run/sync_debug.sh` fetches `origin/debug` and fast-forwards this checkout when `HEAD` is an ancestor of that tip. It then looks at the commits since `/SLURM_shared/st_mdp26/actions-runner/sync-debug.sha`. For every added or modified `next_run/*.sh` other than `sync_debug.sh`, it runs `sbatch` on that file. The SLURM job id is appended to `/SLURM_shared/st_mdp26/actions-runner/sync-debug.log`.

Two passes cannot overlap: the script takes `/SLURM_shared/st_mdp26/actions-runner/sync-debug.lock`. If `sbatch` fails, the saved tip stays put and the next minute retries the scripts from that fetch. A retry submits them again, including any that already succeeded.

If `requirements.txt` or `pyproject.toml` changed in the same range, the pass runs `pip install -r requirements.txt` and `pip install -e .` in `.venv` before `sbatch`.

Pushes that only publish `jobs_logs/`, `runs/`, or `dataset/`, including those from `scripts/push_debug_logs.py`, fast-forward the checkout and do not start a job. See [debug_sync.md](debug_sync.md). That publisher also does not start the runs; it only pushes the tree.

## When the fast-forward does not happen

The merge is `--ff-only`. If this checkout has commits that are not on `origin/debug`, or local edits overlap files in the incoming commit, the pass logs a skip and retries on the next minute. It does not reset or stash.

A push made from this same checkout is already applied when cron wakes up, so it does not submit the scripts that commit just added. The submit happens when the commits arrive from GitHub and this checkout is still behind.
