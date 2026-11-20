# Planned runs on adalovelace

An agent working through GitHub commits on `debug`. This machine fetches that branch. A new or changed shell script in `next_run/` is submitted with `sbatch`. A new or changed shell script in `commands/` is executed on the login node, and its output is published back on `debug` through the log sync. A push that touches neither directory does not start a job or a command.

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

`next_run/sync_debug.sh` fetches `origin/debug` and fast-forwards this checkout when `HEAD` is an ancestor of that tip. It compares that tip with `/SLURM_shared/st_mdp26/actions-runner/sync-debug.sha`. After the fast-forward (and any dependency install), the pass runs added or modified `commands/*.sh` on the login node immediately, then submits added or modified `next_run/*.sh` with `sbatch`, except `sync_debug.sh`. Commands still run when the range has only `commands/` changes and no `next_run/` scripts. A rename counts as a change. A deletion does not run the removed file. The SLURM job id and the command exit code are appended to `/SLURM_shared/st_mdp26/actions-runner/sync-debug.log`.

It does not matter which clone created the commit. A script committed on this checkout runs once that commit is on `origin/debug` and the saved tip is still behind it.

Two passes cannot overlap: the script takes `/SLURM_shared/st_mdp26/actions-runner/sync-debug.lock`. Commands do not wait for `sbatch` success. Tip policy: advance the saved tip only after every `sbatch` in that pass succeeds. If `sbatch` fails, the tip stays put so the next minute retries `next_run` submits; command transcripts already written for this `HEAD` commit are left in place, and those commands are skipped on retry (`jobs_logs/command-<name>-<12-char-commit>.log` already exists). Batch scripts that already queued are submitted again. Command non-zero exits do not block tip advance or `sbatch`.

If `requirements.txt` or `pyproject.toml` changed in the same range, the pass runs `pip install -r requirements.txt` and `pip install -e .` in `.venv` before commands and `sbatch`.

Pushes that only publish `jobs_logs/`, `runs/`, or `dataset/`, including those from `scripts/push_debug_logs.py`, fast-forward the checkout and do not start a job or a command. See [debug_sync.md](debug_sync.md). That publisher also does not start the runs; it only pushes the tree.

## Commands

`commands/` is for a shell command whose output should come back on `debug`. Commit `commands/<name>.sh` as a bash script, directly in that directory. It is not a SLURM job: it has no `#SBATCH` header and it runs on the login node, in `/SLURM_shared/st_mdp26/open-enricher`, with `.venv/bin` on `PATH` and the ffmpeg libraries on `LD_LIBRARY_PATH`. GPU experiments stay in `next_run/`.

The watcher runs each added or changed `commands/*.sh` once on the login node immediately after a successful fetch/fast-forward (and deps install), before any GPU `sbatch` of `next_run/*.sh`. Standard output and standard error go to `jobs_logs/command-<name>-<12-char-commit>.log`. The commit in the file name is the checkout `HEAD` after the fast-forward (the tip being applied). The transcript starts with `script`, `commit`, and `started`, and ends with `exit` and `finished`. Exit 124 means the command was stopped after 15 minutes (`timeout` 900s). A non-zero exit is still published; the command is not run again for the same tip transcript, and a later commit must change the script (new tip / new log name) to re-run it.

Those `command-*.log` files must remain trackable: `.gitignore` has a global `*.log` rule, with a narrow exception `!jobs_logs/command-*.log`. Without that exception, `git status` never shows the transcripts, so `scripts/push_debug_logs.py` prints "jobs_logs and runs are unchanged" and never publishes them. SLURM jobs use `.out`/`.err` and are unaffected. After a merge that adds the exception, verify with `git check-ignore jobs_logs/command-<name>-<12-char-commit>.log` (no output, exit 1 = trackable; `git status` should show `??`). Existing files that were created while ignored may need one `git add -f` (or a re-run that writes a new transcript) before the next `push_debug_logs` pass picks them up.

`scripts/push_debug_logs.py` is already running on this login node. It treats that transcript like any other file under `jobs_logs/` and pushes it to `origin/debug`. While a SLURM job from `jobs_logs/` is queued or running, the push waits up to 10 minutes. Otherwise the next poll, about 30 seconds, pushes it. The agent reads the transcript on `debug` after that push.

A command must not edit files under `commands/` or `next_run/`. The following fetch would see those scripts as changed and execute them again.

## When the fast-forward does not happen

The merge is `--ff-only`. If this checkout has commits that are not on `origin/debug`, or local edits overlap files in the incoming commit, the pass logs a skip and retries on the next minute. It does not reset or stash. The saved tip stays where it is, so the scripts in the incoming commits are still pending.
