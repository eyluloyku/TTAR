#!/bin/bash
#SBATCH --job-name=depthlm_diag
#SBATCH --account=cil_jobs
#SBATCH --gpus=5060ti:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#
# Anchor-quality diagnostic for DepthLM on CIL val data.
#
# Tests two hypotheses suggested by the catastrophic per-image deltas in
# outputs/diagnostics/depthlm_tail10_subset/:
#   (1) Pixtral-12B was trained at funi=750 px → 560x560 inputs are out-of-dist.
#   (2) GRPO <think>/<answer> prompt is wrong; SFT plain prompt is correct.
#
# Default run: 25 images × 15 anchors × 4 conditions = 1500 queries
# (~20-40 min at batch=4 on the 5060-ti).
#
#   sbatch slurm/diagnose_depthlm_quality.sbatch \
#       --out outputs/diagnostics/depthlm_quality/

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PREDS_DIR="${PREDS_DIR:-outputs/val_preds/vits_lora_r8/}"
TRAIN_ROOT="${TRAIN_ROOT:-/cluster/courses/cil/monocular-depth-estimation/train}"

python scripts/diagnose_depthlm_quality.py \
    --preds-dir "$PREDS_DIR" \
    --train-root "$TRAIN_ROOT" \
    "$@"
