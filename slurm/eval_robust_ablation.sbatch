#!/bin/bash
#SBATCH --job-name=vlm_ablate
#SBATCH --account=cil_jobs
#SBATCH --gpus=5060ti:1
#SBATCH --time=14:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#
# Single-pass 6-row ablation:
#   A0 identity  | A1 raw VLM  | A2 calib  | A3 calib+screen
#   A4 robust (calib + screen + gate)  | A5 oracle (GT patch=15)
#
# One VLM forward per image, cached to <out-dir>/vlm_cache/. Re-runs are
# free of VLM cost (cache hit) and only re-do the corrector solves.
#
# Default: 200 stratified val images at K=50, ~7–10h.
#
#   sbatch slurm/eval_robust_ablation.sbatch \
#       --out-dir outputs/diagnostics/ablation_v1/
#
# Tail-10 sanity (~10 min, hits the same cache later):
#   sbatch slurm/eval_robust_ablation.sbatch \
#       --out-dir outputs/diagnostics/ablation_tail10/ \
#       --stems-file outputs/diagnostics/tail10_stems.txt --K 50

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

python scripts/eval_robust_ablation.py \
    --preds-dir "$PREDS_DIR" \
    --train-root "$TRAIN_ROOT" \
    "$@"
