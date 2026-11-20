# Automatic sync to the debug branch

A process on the login node publishes this working tree to `origin/debug` while experiments run. Copying archives to a PC with `rsync` is a different path: see [sync_outputs.md](sync_outputs.md).

The program is `scripts/push_debug_logs.py`. It does not switch branches. The checkout has to stay on `debug`. It runs on the login node (`adalovelace`), where `jobs_logs/`, `runs/`, and the GitHub credentials are visible. A SLURM compute node is the wrong place: it often cannot reach GitHub.

## When it pushes

A pass looks at `jobs_logs/` and `runs/`.

- If nothing there changed, it does nothing. Edits outside those two directories wait for the next log or run change.
- If a SLURM job named in `jobs_logs/` is still queued or running, it commits and pushes at most once every 10 minutes. A log that grows every second does not produce a push every second.
- If that job has left the queue, the next poll pushes immediately. With the loop below, that is within about 30 seconds.
- If `runs/` changed and no job from `jobs_logs/` is still in the queue, it pushes on the next poll.

The job id is the number in the SLURM log name, as in `open-enricher-122096.out`. The script asks `squeue -u $USER` which of those ids are still present. A failure of `squeue` aborts the pass. It does not treat a controller outage as "every job finished".

## What the commit contains

The commit is the whole tree, not only the file that woke the pass: `jobs_logs/`, `runs/`, and any other project file that is not ignored. The message is `Sync job logs and runs`.

Left out:

- `.vscode/`
- `.github/`, because this token cannot push a workflow file and GitHub then rejects the whole push
- `.env`, `.env.*`, and `config/secrets.toml`
- paths already in `.gitignore`, including `runs/local/_cache/` (audio and frame cache)

Commits are authored as Dario Napolitano `<darionapo2.0@gmail.com>` for that command only. Git config is not modified. `git push` uses the `gh` credential helper. The token is a fine-grained personal access token limited to this repository, with Contents set to read and write. It can push every branch of this repo, including `main`. It cannot see other repositories.

One instance runs at a time. The lock is `.git/debug-log-sync.lock`. The last successful push is recorded in `.git/debug-log-sync.json`. Both live under `.git/` and are not committed. If the push fails, the same commit is retried on the next pass and the 10 minute window does not move.

## The running process

Start it from the repository root and leave it attached to a session that survives logout (`tmux`, or the `nohup` line below):

```bash
nohup python3 scripts/push_debug_logs.py --loop --poll-seconds 30 >> .git/debug-log-sync.out 2>&1 &
```

Confirm it:

```bash
ps -u "$USER" -o pid,etime,cmd | grep push_debug_logs
```

A single pass, suitable for cron once a minute, is `python3 scripts/push_debug_logs.py --once`. `--dry-run` prints the decision and does not commit.

Stopping the process stops the sync. Closing the terminal that owns it does the same, unless it was started under `nohup` or `tmux`. After a reboot it has to be started again. Code changes to the script apply only to processes started after the change.
