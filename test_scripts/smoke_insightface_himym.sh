#!/bin/bash
#SBATCH --job-name=smoke-insightface-himym
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/SLURM_shared/st_mdp26/open-enricher/jobs_logs/smoke-insightface-himym-%j.out
#SBATCH --error=/SLURM_shared/st_mdp26/open-enricher/jobs_logs/smoke-insightface-himym-%j.err
#SBATCH --account=studenti
#SBATCH --partition=all-nodes

# InsightFace buffalo_l face smoke for How_I_Met_Your_Mother.S01E02.mkv only.
# onnxruntime-gpu 1.26 loads the CUDA 12 libraries shipped with torch.

set -euo pipefail

cd /SLURM_shared/st_mdp26/open-enricher
export PATH="/SLURM_shared/st_mdp26/open-enricher/.venv/bin:$PATH"

FFMPEG_ROOT=/SLURM_shared/st_mdp26/ffmpeg-libs
SITE=/SLURM_shared/st_mdp26/open-enricher/.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$SITE/cudnn/lib:$SITE/cublas/lib:$SITE/cuda_runtime/lib:$SITE/cufft/lib:$SITE/curand/lib:$SITE/cusolver/lib:$SITE/cusparse/lib:$SITE/cuda_nvrtc/lib:$SITE/nvjitlink/lib:$FFMPEG_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

VIDEO="How_I_Met_Your_Mother.S01E02.mkv"

srun --unbuffered \
  /SLURM_shared/st_mdp26/open-enricher/.venv/bin/python \
  test_scripts/insightface_faces.py \
  --video "dataset/video_inputs/${VIDEO}" \
  --sample-fps 1
