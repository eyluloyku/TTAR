#!/bin/bash
#SBATCH --job-name=vlm_robust
#SBATCH --account=cil_jobs
#SBATCH --gpus=5060ti:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#
# Robust DepthLM-anchored corrector (changes 1+5):
#   - per-image median-bias scale alignment of VLM outputs
#   - hard + adaptive MAD outlier screening on calibrated anchor deltas
#   - per-image agreement gate (ρ(log d_VLM, log D0) ≥ τ to refine, else D0)
#
# Tail-10 first (10 images, ~10–15 min), then full val once we trust the knobs.
#
#   sbatch slurm/eval_robust_vlm_refine.sbatch \
#       --out-dir outputs/diagnostics/vlm_robust_tail10/ \
#       --stems-file outputs/diagnostics/one_tail_stem.txt
#
#   sbatch slurm/eval_robust_vlm_refine.sbatch \
#       --out-dir outputs/diagnostics/vlm_robust_val/

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

python scripts/eval_robust_vlm_refine.py \
    --preds-dir "$PREDS_DIR" \
    --train-root "$TRAIN_ROOT" \
    "$@"
