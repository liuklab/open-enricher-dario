# open-enricher

Experiment library for video understanding. This repository holds reusable Python code for inference tasks such as textual metadata extraction and video embeddings.

Run inference on Google Colab Pro from your local terminal with [Google Colab CLI](https://github.com/googlecolab/google-colab-cli). The library lives in `src/open_enricher/`. Thin scripts under `scripts/` are the supported runners. Notebooks are optional leftovers for inspecting a mounted session.

The first bootstrap uses [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) open weights, with videos stored on Google Drive.

## What this covers

- **Metadata:** prompt a vision-language model and parse JSON-like text (title guess, summary, tags, objects, scenes).
- **Embeddings:** run the same model and mean-pool last hidden states into one vector per video. This is an experimental pooled representation, not a dedicated embedding checkpoint.

Training, retrieval indexes, and film-specific schemas are out of scope for this bootstrap.

## Layout

```text
src/open_enricher/    library code
scripts/              Colab CLI runners (supported)
notebooks/            optional inspection notebooks
config.toml           non-secret project settings
.env.example          secret template (copy to .env)
```

## Colab CLI setup

The CLI is a local tool (Linux or macOS, including WSL). Native Windows is not documented as supported by Google Colab CLI; use WSL or run the PowerShell wrapper if `colab` is on PATH. Do not add the CLI to this project's `requirements.txt`.

```bash
uv tool install google-colab-cli
# or: pip install google-colab-cli
```

Authenticate once with Application Default Credentials. The CLI needs the Colab and Cloud Platform scopes; a generic Google login that only grants profile/email will fail with "Scope has changed".

```bash
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
```

Then use `colab --auth=adc ...` (the bootstrap scripts already pass `--auth=adc`). Do not use a personal `~/.colab-cli-oauth-config.json` unless it is Google's bundled Cloud SDK client; a custom OAuth client typically cannot grant the `colaboratory` scope.

1. Copy `.env.example` to `.env` and fill in `HF_TOKEN` and `GITHUB_TOKEN` only. Do not commit `.env`. Paths, git remote, model, GPU, and timeouts live in [`config.toml`](config.toml).
2. Put videos under the Drive folder in `config.toml` (`[drive] data`). Any nested layout is fine; the library recursively finds `.mp4`, `.mkv`, `.avi`, `.mov`, and `.webm`.
3. Commit and `git push` the code you want the VM to run. The VM clones `[git] url` into `[git] repo_dir`. Drive is not used for source code.
4. From the local checkout, start a GPU session, mount Drive for videos, sync the library via git, and run:

```bash
colab --auth=adc new -s enricher --gpu A100
colab --auth=adc drivemount -s enricher
colab --auth=adc install -s enricher -r requirements.txt
colab --auth=adc exec -s enricher --timeout 600 -f scripts/install_package.py
colab --auth=adc exec -s enricher --timeout 7200 -f scripts/run_metadata.py
# colab --auth=adc stop -s enricher
```

The bootstrap scripts load `.env` (tokens) and `config.toml` (everything else), then inject those values into the Colab kernel before running install/inference. This Colab CLI build has no `exec --env` flag. Do not use Colab Secrets / userdata; they do not work in CLI sessions.

Or run the same sequence with a helper script:

```bash
# Linux, macOS, WSL
./scripts/bootstrap_colab.sh
./scripts/bootstrap_colab.sh embeddings
```

```powershell
# Windows PowerShell (requires `colab` on PATH)
.\scripts\bootstrap_colab.ps1
.\scripts\bootstrap_colab.ps1 embeddings
```

From `cmd.exe` use `scripts\bootstrap_colab.cmd`. If `colab` is missing on Windows, run the bash script inside WSL instead.

`colab exec -f` ships only that one local file to the remote kernel. `scripts/install_package.py` clones or updates the repo on the VM and runs `pip install -e`. Unpushed local edits are not visible on the VM. The bootstrap scripts read `config.toml` plus `.env` and pass git URL, ref, tokens, and data paths automatically.

If `[git] url` / `[git] ref` are empty, bootstrap fills them from `git remote get-url origin` and the current branch.

For a private remote, set `GITHUB_TOKEN` (or `GIT_TOKEN`) in `.env`; install_package injects it into the HTTPS URL and does not print it.

The CLI default `--timeout` is about 30 seconds. Model download and video inference need a much larger value (the commands above use 7200).

GPU examples: `T4`, `L4`, `A100`. Set `[colab] gpu` in `config.toml`. `colab drivemount` may open a browser once for Google OAuth.

To pick a video or a larger model, set `[drive] video` and `[model] id` in `config.toml`.

If a video backend error appears, install `ffmpeg` on the VM or add `torchcodec` on top of the Colab GPU PyTorch install.

PyTorch itself should come from the Colab GPU image. `requirements.txt` only adds Transformers and related extras.

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
