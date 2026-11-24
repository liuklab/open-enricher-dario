# Running a local experiment

Run from the main checkout with its Python environment active. Place original
videos in `dataset/video_inputs/` and reusable model files in
`dataset/models_cache/`. Configure the optional encoders before running a
variant that uses them. [Encoder setup](encoders.md)

```bash
python -m open_enricher --list
python -m open_enricher --only 12-combined --video Suits.S01E02.mkv
sbatch run_job.sh --only 10-character-graph --video Suits.S01E02.mkv
```

Submit from the repository root. The Slurm launcher passes its arguments to
the same CLI. With no arguments it runs `12-combined` on the available titles.
Set `OPEN_ENRICHER_PYTHON` to choose an existing worker interpreter and
`FFMPEG_ROOT` for an FFmpeg installation with `bin/` and `lib/` directories.
On the original cluster the launcher also detects the adjacent `ffmpeg-libs/`
installation and puts the worker interpreter directory on `PATH`. On Slurm, stdout and stderr use the scheduler's job log.

## Choose which evidence to collect

The packaged variant JSONs specify frame budgets, optional stages and checkpoint
overrides. `09-bassl-uniform` and `09-bassl-adaptive` use scene boundaries,
`10-character-graph` enables face and speaker evidence, `11-visual-memory`
enables frame retrieval, and `12-combined` enables all three. Use
`OPEN_ENRICHER_VARIANTS_DIR` to point at a directory of your own variant JSONs
and select one with `--only`. Check the episode manifest for the stages that
actually succeeded and inspect their outputs during comparison.

## Output and cache paths

| Default path | Contents |
| --- | --- |
| `dataset/video_inputs/` | Original videos |
| `dataset/models_cache/` | Reusable model weights and encoder runtimes |
| `dataset/outputs/` | Working stage outputs |
| `runs/local/<experiment_id>/<video>/` | Archived stage files for this run |
| `runs/local/_cache/` | Reusable stage data and sampled-frame evidence |
| `runs/local/progress.jsonl` | Run progress |
| `local-archive/` | Read-only-in-practice historical results, local to the cluster |

For an independent run, select both a new working-output root and an empty
run root. The dataset root still supplies the videos and model cache.

```bash
export OPEN_ENRICHER_OUTPUT_ROOT="$PWD/dataset/outputs/my-comparison"
export OPEN_ENRICHER_RUNS_ROOT="$PWD/runs/my-comparison"
python -m open_enricher --only 12-combined --video Suits.S01E02.mkv
```

Use `--force` when intentionally repeating a completed variant. Name a separate
run root to retain both results. The built-in evaluator writes separate scores
after inference. [Evaluation](evaluator.md)
