# open-enricher

Experiment library for video understanding. This repository holds reusable Python code for textual metadata extraction and a modular remote pipeline of interchangeable stages.

Run one-shot metadata inference on Google Colab from your local terminal with [Google Colab CLI](https://github.com/googlecolab/google-colab-cli), or on Kaggle. For an ordered pipeline (audio, color, metadata) over HTTP, deploy the FastAPI worker and dispatch stages from the laptop.

The first bootstrap uses [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) open weights, with videos stored on Google Drive (Colab) or in a flat dataset folder.

## What this covers

- **Metadata:** prompt a vision-language model and write the generated text as-is.
- **Remote pipeline:** run ordered stages (`audio`, `color`, `metadata`) on a worker reached over Tailscale. Each stage is a separate `POST /stages/{name}`.

Missing config, a missing video file, or a processor that does not emit video tensors raises. There is no fallback to CPU, to the first video in the folder, or to Hugging Face `auto` dtype/device_map.

## Layout

```text
src/open_enricher/       library, pipeline stages, FastAPI worker, HTTP client
colab_cli/               Colab CLI one-shot metadata bootstrap and API deploy
colab_notebook/          one-cell Colab session bootstrap (no CLI)
remote_host/             scp/ssh deploy to a Tailscale host
kaggle_job/              optional Kaggle metadata launcher
scripts/                 shared metadata runner and local evaluation
.env.example             secret template (copy to .env)
```

## Secrets and paths

1. Copy `.env.example` to `.env` and fill in `HF_TOKEN`. For Colab CLI / scp worker deploys that should join Tailscale, also set `TS_AUTH_KEY` or `TAILSCALE_KEY`. Do not commit `.env`.
2. Paths and model live as constants in [`src/open_enricher/config.py`](src/open_enricher/config.py). Metadata generation params live in [`src/open_enricher/metadata_params.json`](src/open_enricher/metadata_params.json).
3. Put videos directly in the folder named by `DRIVE_DATA`. Subfolders are ignored. Supported extensions: `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`. The API and runners resolve a **file name** with `resolve_video`, not a path.

## Remote pipeline

Install the remote extra on the machine that talks to the worker (and on the worker itself when using the deploy scripts):

```bash
pip install -e ".[remote]"
```

Stage order is fixed in [`src/open_enricher/pipeline/order.py`](src/open_enricher/pipeline/order.py): `audio`, `color`, `metadata`. The local orchestrator walks that list and waits for each response:

```bash
python -m open_enricher.pipeline \
  --host 100.x.y.z \
  --port 8000 \
  --video Stranger_Things.S01E03.mkv
```

`GET /stages` lists the order. `POST /stages/{name}` body is `{"video": "file.mkv"}`. The worker resolves the name under `DRIVE_DATA`.

### Deploy the worker

Three ways to put code on the worker. Frequent stage edits should use the local checkout tarball (CLI or scp), not a notebook `git pull` loop.

**Colab CLI** (preferred when the CLI is available):

```bash
python colab_cli/deploy_server.py \
  --session enricher-api \
  --gpu T4 \
  --auth adc \
  --install-timeout 600

# later code refreshes
python colab_cli/deploy_server.py \
  --session enricher-api \
  --gpu T4 \
  --auth adc \
  --install-timeout 600 \
  --skip-setup
```

**Tailscale host** (SSH already works on the tailnet):

```bash
python remote_host/deploy.py \
  --host 100.x.y.z \
  --user ubuntu \
  --remote-dir /home/ubuntu/open-enricher
```

**Colab notebook, no CLI:** paste [`colab_notebook/session.py`](colab_notebook/session.py) into one cell. Set `BRANCH` at the top. Provide Colab userdata secrets `GITHUB` and `TAILSCALE_KEY`. The cell mounts Drive, clones or pulls, creates a venv once, connects Tailscale, and runs the server in the foreground.

The bootstrap only prepares the runtime. Stages run when the orchestrator POSTs.

## Colab CLI one-shot metadata

The CLI is a local tool (Linux or macOS, including WSL). Native Windows is not supported; use WSL. Do not add the CLI to this project's `requirements.txt`.

```bash
uv tool install google-colab-cli
```

Authenticate once with Application Default Credentials:

```bash
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
```

```bash
python colab_cli/bootstrap.py \
  --session enricher \
  --gpu T4 \
  --auth adc \
  --timeout 7200 \
  --install-timeout 600
```

The bootstrap packs `src/`, `pyproject.toml`, `requirements.txt`, and `.env` (if present) into one archive and uploads it with `colab upload`. Unpushed local edits are included. The VM does not clone from GitHub.

`colab exec -f` ships only one local file to the remote kernel. [`colab_cli/install_local.py`](colab_cli/install_local.py) extracts `/content/open-enricher.tgz` into `/content/open-enricher` and runs `pip install -e`. Hugging Face reads `HF_TOKEN` from the `.env` that landed in that checkout.

Do not use Colab Secrets / userdata for CLI sessions; they do not work there.

The CLI default `--timeout` is about 30 seconds. Model download and video inference need a much larger value; pass `--timeout` and `--install-timeout` explicitly.

GPU examples: `T4` (Colab free), `L4`, `A100`. Pass `--gpu`. `colab drivemount` may open a browser once for Google OAuth.

To pick a video or a larger model, set `DRIVE_VIDEO` and `MODEL_ID` in `src/open_enricher/config.py`. `METADATA_VIDEO_SECONDS` is `None` for the whole file, or a positive integer for only the first that many seconds. On T4 set `MODEL_DTYPE` to `float16`. Use `bfloat16` only on Ampere GPUs (`L4`, `A100`). `MODEL_DEVICE_MAP` is `cuda`: inference does not fall back to CPU.

PyTorch itself should come from the Colab GPU image. `requirements.txt` only adds Transformers and related extras.

```bash
colab --auth=adc stop -s enricher
```

## Kaggle metadata

```bash
python kaggle_job/launch.py
```

Videos come from the dataset named in [`dataset/dataset-metadata.json`](dataset/dataset-metadata.json). The launcher packs this checkout and runs [`scripts/run.py`](scripts/run.py) on the kernel.

## Automatic metadata evaluation

[`scripts/evaluate.py`](scripts/evaluate.py) runs metadata inference on Colab via [`colab_cli/bootstrap.py`](colab_cli/bootstrap.py), then uses a local [Cursor SDK](https://cursor.com/docs/sdk/python) agent (`composer-2.5`) to judge the output and rewrite [`src/open_enricher/metadata_params.json`](src/open_enricher/metadata_params.json) in full.

```bash
pip install -e ".[eval]"
```

Put `CURSOR_API_KEY=...` in repo-root [`.env.eval`](.env.eval). Bootstrap only uploads `.env` (with `HF_TOKEN`), never `.env.eval`.

```bash
python scripts/evaluate.py \
  --max-iters 3 \
  --session enricher \
  --gpu T4 \
  --auth adc \
  --timeout 7200 \
  --install-timeout 600
```

## Local install

```bash
pip install -e .
pip install -e ".[eval]"
pip install -e ".[remote]"
```

## Switch model

Default checkpoint: `Qwen/Qwen3-VL-8B-Instruct`. Change `MODEL_ID` in `src/open_enricher/config.py`, or build a full `ModelConfig`:

```python
from open_enricher import ModelConfig, load_model

model_config = ModelConfig(
    model_id="Qwen/Qwen3-VL-8B-Instruct",
    dtype="float16",
    device_map="cuda",
    attn_implementation="sdpa",
)
model, processor = load_model(model_config)
```

Keep `fps` in `metadata_params.json` low unless you know the runtime can hold more frames.

## Library usage

```python
from open_enricher import (
    DRIVE_CONFIG,
    METADATA_CONFIG,
    MODEL_CONFIG,
    extract_metadata,
    load_model,
    resolve_video,
    save_json,
    save_text,
)

model, processor = load_model(MODEL_CONFIG)
video = resolve_video(DRIVE_CONFIG.video, DRIVE_CONFIG.root)

text = extract_metadata(model, processor, video, METADATA_CONFIG)
save_text(f"{DRIVE_CONFIG.output_dir}/metadata/{video.stem}.txt", text)
save_json(
    f"{DRIVE_CONFIG.output_dir}/metadata/{video.stem}.json",
    {"video": str(video), "model_id": MODEL_CONFIG.model_id},
)
```
