# Open-Enricher

Client-server pipeline for video understanding. A laptop packs the code and starts a GPU worker. The worker runs five stages and writes structured metadata for a recommender:

`audio` (WhisperX) → `color` (scene colors) → `vlm` (Qwen3-VL, in 5-minute windows) → `llm` (Qwen3-8B profile) → `judger` (deterministic score).

The worker can run on Google Colab or on a Kaggle GPU kernel. Both modes keep the same stage API. The laptop calls that API. It does not need a local GPU.

## Install the client

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-cli.txt
```

Create `.env` in the repository root:

```env
HF_TOKEN=hf_your_token
```

The token is uploaded to the worker and is not stored in git. Pyannote diarization weights are gated. If the token cannot download them, the audio stage still saves the transcript and marks every speaker as `UNKNOWN`.

## Colab

Requires the `colab` CLI on `PATH`, already authenticated.

```bash
python deploy.py --mode colab-cli --target enricher --gpu T4
python -m src.open_enricher.client \
  --mode colab-cli --target enricher \
  --video Stranger_Things.S01E03.mkv \
  --stages audio color vlm llm judger \
  --timeout 10800
```

`deploy.py` reuses the named session when it already exists. Otherwise it starts a T4 session and mounts Google Drive. The data lake is:

```text
/content/drive/MyDrive/Open-Enricher-Database/
├── Models-Cache/
├── Video-Inputs/          # put the source video here
└── Outputs/
    ├── Audio-Results/
    ├── Color-Results/
    ├── VLMs-Results/
    ├── LLMs-Results/
    └── Judger-Results/
```

The client opens an SSH tunnel to port 8000, runs the requested stages, and saves each JSON response under `outputs/` on the laptop. The full artifacts stay on Drive.

## Kaggle

Requires the `kaggle` CLI and `kaggle auth login`.

Put the source videos in a private dataset and record its id in `dataset/dataset-metadata.json`:

```json
{
  "title": "open-enricher-dataset",
  "id": "your-user/open-enricher-dataset",
  "licenses": [{"name": "CC0-1.0"}]
}
```

Videos must sit at the root of that dataset. Kaggle mounts it at `/kaggle/input/<slug>`.

```bash
python deploy.py --mode kaggle --target open-enricher --gpu T4
python -m src.open_enricher.client \
  --mode kaggle --target open-enricher \
  --video Stranger_Things.S01E03.mkv \
  --stages audio color vlm llm judger \
  --timeout 10800
```

`--target` is the kernel slug. The full kernel id is `<kaggle-user>/open-enricher`.

Deploy uploads a private code dataset (`<user>/open-enricher-code`, including `.env`), pushes the kernel on a T4 with internet enabled, and waits until the worker prints a tunnel URL. Kaggle kernels do not accept inbound SSH, so the kernel opens an outbound Cloudflare tunnel and the client calls the same HTTP stage API through that URL. The URL is cached in `.session/<slug>.url`.

Outputs and the model cache are written under `/kaggle/working/Open-Enricher-Database`. Pull them while the kernel is still running:

```bash
kaggle kernels output <user>/open-enricher -p outputs
```

The kernel stays up for 8 hours after deploy so later stage commands can reuse it.

## Stages

| Stage | What it writes |
| :--- | :--- |
| `audio` | Timestamped transcript |
| `color` | Keyframes and dominant hex colors |
| `vlm` | One description per 5-minute window, merged into one JSON. A local 8B model summarizes the transcript first and that summary is added to the prompt, together with the color digest. |
| `llm` | `VideoProfile` for search, audience, and a short summary. It reads the brief instead of the full transcript so a 15 GB GPU can hold the model. |
| `judger` | Deterministic score of that profile. No second model. |

Run a subset by passing `--stages`. Later stages reuse JSON already stored in the data lake.

## Layout

```text
deploy.py                      # upload and start the worker
deployment/launch.sh           # remote install and uvicorn
deployment/kaggle_worker.py    # Kaggle kernel entrypoint
src/open_enricher/client/      # laptop CLI
src/open_enricher/remote/      # FastAPI worker and stages
```

`old_files/` and the per-experiment notes from the first Colab round are not part of this branch.
