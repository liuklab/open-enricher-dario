#!/usr/bin/env bash
#SBATCH --job-name=open-enricher
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --account=studenti
#SBATCH --partition=all-nodes
#SBATCH --gres=gpu:1

set -euo pipefail
repo_dir=${SLURM_SUBMIT_DIR:-$(dirname "$(realpath "$0")")}
cd "$repo_dir"
python_bin=${OPEN_ENRICHER_PYTHON:-"$repo_dir/.venv/bin/python"}
ffmpeg_root=${FFMPEG_ROOT:-"$repo_dir/../ffmpeg-libs"}
if [[ -d "$ffmpeg_root/lib" ]]; then
    export PATH="$ffmpeg_root/bin:$PATH"
    export LD_LIBRARY_PATH="$ffmpeg_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export PATH="$(dirname "$python_bin"):$PATH"
if (( $# == 0 )); then
    set -- --only 12-combined
fi
exec srun --unbuffered "$python_bin" -m open_enricher "$@"
