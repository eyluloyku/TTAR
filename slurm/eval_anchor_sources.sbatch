#!/bin/bash
#SBATCH --job-name=anchors
#SBATCH --account=cil_jobs
#SBATCH --gpus=5060ti:1
#SBATCH --time=05:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#
# Backend-agnostic anchor-source eval. Mostly CPU work; GPU requested only
# to land on the 5060-ti env.
#
# Oracle backend (sanity — should reproduce Phase 1c numbers within seed
# variance for the random-anchor cells):
#   sbatch slurm/eval_anchor_sources.sbatch \
#       --backend oracle \
#       --out-dir outputs/diagnostics/anchors_oracle/
#
# Second-model backend (precomputed predictions):
#   sbatch slurm/eval_anchor_sources.sbatch \
#       --backend second \
#       --second-preds-dir outputs/val_preds/marigold/ \
#       --out-dir outputs/diagnostics/anchors_marigold/
#
# VLM backend (after Phase 5 lands):
#   sbatch slurm/eval_anchor_sources.sbatch \
#       --backend vlm \
#       --out-dir outputs/diagnostics/anchors_vlm/

set -euo pipefail

. /etc/profile.d/modules.sh
module load cuda/12.6.0

source /cluster/courses/cil/envs/etc/profile.d/conda.sh

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
if [[ "${GPU_NAME}" == *"1080"* ]]; then
    CIL_ENV="${CIL_ENV:-monocular-depth-estimation}"
else
    CIL_ENV="${CIL_ENV:-monodepth-5060}"
fi
echo "[slurm] GPU: $GPU_NAME"
echo "[slurm] conda env: $CIL_ENV"

conda activate "/cluster/courses/cil/envs/envs/${CIL_ENV}"

export PYTHONUSERBASE="${HOME}/.local"
export HF_HOME="${HF_HOME:-/work/scratch/$USER/hf_cache}"
export PYTHONUNBUFFERED=1   # flush print() immediately so logs show live progress
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PREDS_DIR="${PREDS_DIR:-outputs/val_preds/vits_lora_r8/}"
TRAIN_ROOT="${TRAIN_ROOT:-/cluster/courses/cil/monocular-depth-estimation/train}"

python scripts/eval_anchor_sources.py \
    --preds-dir "$PREDS_DIR" \
    --train-root "$TRAIN_ROOT" \
    "$@"
