# Running a local experiment

Run from the main checkout with its Python environment active. Place original
videos in `dataset/video_inputs/` and reusable model files in
`dataset/models_cache/`. Configure the optional encoders before running a
variant that uses them. [Encoder setup](encoders.md)

```bash
python -m open_enricher --list
python -m open_enricher --only 12-combined --video suits_s01e02.mkv
sbatch run_job.sh --only 10-character-graph --video suits_s01e02.mkv
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

`13-narrate-exchanges` uses 4-minute windows and up to 40 current frames.
In the same call as the interval summary, the dialogue model selects up to two
short exchanges by transcript line ID, most informative first. The VLM sees
only the current frames. The narrator receives the local summary and selected
lines, plus the summaries and top exchange from each of the last three windows.
In `*_dialogue.json` and `*_vlm.json`, each window's `dialogue_exchanges` field
contains the original ASR text, timestamps, speaker labels and window-local IDs.
A silent window or an interval without a valid selection has an empty list.

Each window also has JPEGs in
`runs/local/_cache/<video>/frames/13-narrate-exchanges/scene_XX/`.
The archived `<video>_vlm_frames.json` lists each image, its clip index and its
timestamp in the original video. The exporter uses Qwen's sampler on the same
clip. The JPEGs show the selected images before the model's spatial resizing
and normalization. Short final windows may contain fewer than 40 frames.
Copy the frame directory alongside the manifest when keeping a run.

```bash
python -m open_enricher --only 13-narrate-exchanges --video suits_s01e02.mkv
```

WhisperX transcribes the selected audio track in its spoken language and saves
the language code in `*_audio_language.json`. Italian audio yields Italian plot
summaries, scene descriptions and free-text RecSys fields. JSON keys and the
controlled labels for primary genre, pacing and target age group stay in English.
Selected ASR lines and on-screen quotations retain their source wording.
The two API evaluations return Italian commentary for all films.
Italian word alignment uses TorchAudio's VoxPopuli checkpoint,
`dataset/models_cache/torch/hub/checkpoints/wav2vec2_voxpopuli_base_10k_asr_it.pt`
with the default cache layout.

Automatic language detection is the default for mixed batches. To force Italian
for a specific film with an ambiguous opening, run it on its own:

```bash
OPEN_ENRICHER_AUDIO_LANGUAGE=it python -m open_enricher --only 13-narrate-exchanges --video il_vigile.avi
```

WhisperX uses FFmpeg's default audio selection. For a multilingual container,
verify that the desired dub is the selected track. The language override tells
WhisperX how to transcribe that track. On archived inputs without a recorded
language, downstream stages use the override when supplied, or English by default.

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
python -m open_enricher --only 12-combined --video suits_s01e02.mkv
```

Use `--force` when intentionally repeating a completed variant. Name a separate
run root to retain both results. The built-in evaluator writes separate scores
after inference. [Evaluation](evaluator.md)
