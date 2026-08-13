# open-enricher

Experiment library for video understanding. This repository holds reusable Python code for inference tasks such as textual metadata extraction and video embeddings.

Run inference on Google Colab Pro from your local terminal with [Google Colab CLI](https://github.com/googlecolab/google-colab-cli). The library lives in `src/open_enricher/`. `scripts/bootstrap_colab.py` is the supported runner.

The first bootstrap uses [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) open weights, with videos stored on Google Drive.

## What this covers

- **Metadata:** prompt a vision-language model and parse JSON-like text (title guess, summary, tags, objects, scenes).
- **Embeddings:** run the same model and mean-pool last hidden states into one vector per video. This is an experimental pooled representation, not a dedicated embedding checkpoint.

Training, retrieval indexes, and film-specific schemas are out of scope for this bootstrap.

## Layout

```text
src/open_enricher/    library code
scripts/              Colab CLI bootstrap and runners
config.toml           non-secret project settings
.env.example          secret template (copy to .env)
```

## Colab CLI setup

The CLI is a local tool (Linux or macOS, including WSL). Native Windows is not supported; use WSL. Do not add the CLI to this project's `requirements.txt`.

```bash
uv tool install google-colab-cli
# or: pip install google-colab-cli
```

Authenticate once with Application Default Credentials. The CLI needs the Colab and Cloud Platform scopes; a generic Google login that only grants profile/email will fail with "Scope has changed".

```bash
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
```

Then use `colab --auth=adc ...` (the bootstrap script already passes `--auth=adc`). Do not use a personal `~/.colab-cli-oauth-config.json` unless it is Google's bundled Cloud SDK client; a custom OAuth client typically cannot grant the `colaboratory` scope.

1. Copy `.env.example` to `.env` and fill in `HF_TOKEN`. Do not commit `.env`. Paths, model, GPU, and timeouts live in [`config.toml`](config.toml).
2. Put videos under the Drive folder in `config.toml` (`[drive] data`). Any nested layout is fine; the library recursively finds `.mp4`, `.mkv`, `.avi`, `.mov`, and `.webm`.
3. From the local checkout, start (or reuse) a GPU session, mount Drive for videos, upload this checkout as a tarball, and run:

```bash
python scripts/bootstrap_colab.py
python scripts/bootstrap_colab.py embeddings
```

The bootstrap packs `src/`, `pyproject.toml`, `requirements.txt`, `config.toml`, and `.env` (if present) into one archive and uploads it with `colab upload`. Unpushed local edits are included. The VM does not clone from GitHub.

`colab exec -f` ships only one local file to the remote kernel. `scripts/install_local.py` extracts `/content/open-enricher.tgz` into `/content/open-enricher` and runs `pip install -e`. Hugging Face reads `HF_TOKEN` from the `.env` that landed in that checkout.

Do not use Colab Secrets / userdata; they do not work in CLI sessions.

The CLI default `--timeout` is about 30 seconds. Model download and video inference need a much larger value (`[colab] timeout` defaults to 7200).

GPU examples: `T4`, `L4`, `A100`. Set `[colab] gpu` in `config.toml`. `colab drivemount` may open a browser once for Google OAuth.

To pick a video or a larger model, set `[drive] video` and `[model] id` in `config.toml`.

If a video backend error appears, install `ffmpeg` on the VM or add `torchcodec` on top of the Colab GPU PyTorch install.

PyTorch itself should come from the Colab GPU image. `requirements.txt` only adds Transformers and related extras.

The named session stays running after inference. Stop it when you are done:

```bash
colab --auth=adc stop -s enricher
```

## Local install

```bash
pip install -e .
```

This is for editing and importing the library on your machine. Inference is meant to run on the Colab VM.

## Switch model

Default checkpoint: `Qwen/Qwen3-VL-2B-Instruct`.

```python
from open_enricher import ModelConfig, load_model

model_config = ModelConfig(model_id="Qwen/Qwen3-VL-8B-Instruct")
model, processor = load_model(model_config)
```

Or set `[model] id` in `config.toml`. Larger models need more GPU memory. Keep `[model] num_frames` low (default 16) unless you know the runtime can hold more.

## Library usage

```python
from open_enricher import (
    DriveConfig,
    EmbeddingConfig,
    MetadataConfig,
    ModelConfig,
    extract_embedding,
    extract_metadata,
    list_videos,
    load_model,
    save_embedding,
    save_json,
)

drive = DriveConfig(
    root="/content/drive/MyDrive/open_enricher/data",
    output_dir="/content/drive/MyDrive/open_enricher/outputs",
)
model, processor = load_model(ModelConfig())
video = list_videos(drive.root)[0]

meta = extract_metadata(model, processor, video, MetadataConfig(), model_id="Qwen/Qwen3-VL-2B-Instruct")
save_json(f"{drive.output_dir}/metadata/{video.stem}.json", meta)

vector = extract_embedding(model, processor, video, EmbeddingConfig())
save_embedding(
    f"{drive.output_dir}/embeddings/{video.stem}.npy",
    vector,
    sidecar={"video": str(video), "model_id": "Qwen/Qwen3-VL-2B-Instruct", "shape": list(vector.shape)},
)
```
