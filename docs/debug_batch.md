# Debug batch on adalovelace

A push to `origin/debug` updates the checkout on this machine and can submit every experiment variant as one SLURM job. The checkout that moves is `/SLURM_shared/st_mdp26/open-enricher`, the same tree the batch runs from. `dataset/`, `runs/`, `jobs_logs/`, `.env`, and `.venv` stay on disk.

The cluster address is private, so a GitHub-hosted runner cannot reach it. The live trigger is cron on the login node, user `st_mdp26`:

```text
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
* * * * * /SLURM_shared/st_mdp26/open-enricher/next_run/sync_debug.sh >> /SLURM_shared/st_mdp26/actions-runner/sync-debug.log 2>&1
```

Once a minute, `next_run/sync_debug.sh` fetches `origin/debug` and fast-forwards this checkout when `HEAD` is an ancestor of that tip. The last applied commit is `/SLURM_shared/st_mdp26/actions-runner/sync-debug.sha`. A pass that finds the same tip exits without doing anything. Two passes cannot overlap: the script takes `/SLURM_shared/st_mdp26/actions-runner/sync-debug.lock`.

## When a job is submitted

`sbatch next_run/run_experiments.sh` runs only if the new commits touch a path outside these prefixes:

- `jobs_logs/`
- `runs/`
- `dataset/`

Pushes that only publish logs or archives, including those from `scripts/push_debug_logs.py`, fast-forward the checkout and do not start a GPU job. A change under `docs/`, `src/`, `next_run/`, or any other tracked path does start one. If `requirements.txt` or `pyproject.toml` is part of that change, the pass runs `pip install -r requirements.txt` and `pip install -e .` in `.venv` before `sbatch`.

The batch script asks for one GPU, 8 CPUs, 32 GB, and 72 hours on partition `all-nodes`, account `studenti`. The job name is `open-enricher-next`. Stdout and stderr go to `jobs_logs/open-enricher-next-<jobid>.out` and `.err`. Inside the allocation it runs every variant JSON under `src/open_enricher/experiments/variants/`, in the order the experiment loop lists them, on these videos:

- `How_I_Met_Your_Mother.S01E02.mkv`
- `Silicon_Valley.S06E01.mkv`
- `The_Kingdom.S01E02.mkv`

`Suits.S01E02.mkv` is in `dataset/video_inputs/` and is not part of this batch. There is no `--only` filter, so a variant added later is included on the next submit. Submit the same script by hand with:

```bash
sbatch /SLURM_shared/st_mdp26/open-enricher/next_run/run_experiments.sh
```

## When the fast-forward does not happen

The merge is `--ff-only`. If this checkout has commits that are not on `origin/debug`, or local edits overlap files in the incoming commit, the pass logs a skip and retries on the next minute. It does not reset or stash. The log line is in `/SLURM_shared/st_mdp26/actions-runner/sync-debug.log`.

A job already in the queue is left alone. A second qualifying push submits a second `open-enricher-next` job.

## GitHub Action

`.github/workflows/deploy-debug.yml` describes the same sync and `sbatch` on a self-hosted runner labeled `adalovelace`. That file is on this machine and is not on `origin/debug`. The token used for `git push` from here cannot update workflow files, and it cannot register a runner. The runner program is unpacked at `/SLURM_shared/st_mdp26/actions-runner` and is not configured.

Do not turn that workflow on while this cron entry is active. Both would submit a batch for the same push. Remove the cron line first:

```bash
crontab -e
```
