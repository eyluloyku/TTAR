#!/bin/bash
#SBATCH --job-name=bnd_phase0
#SBATCH --account=cil_jobs
#SBATCH --gpus=5060ti:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#
# Stage B v5 — Phase 0 boundary error-mode diagnostic.
# Pure CPU work — GPU requested only to land on the 5060-ti env.
#
# Full val:
#   sbatch slurm/diagnose_boundary_errormode.sbatch \
#       --out-dir outputs/diagnostics/boundary_phase0/

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
echo "[slurm] GPU: $GPU_NAME  conda env: $CIL_ENV"
conda activate "/cluster/courses/cil/envs/envs/${CIL_ENV}"

export PYTHONUSERBASE="${HOME}/.local"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PREDS_DIR="${PREDS_DIR:-outputs/val_preds/vits_lora_r8/}"
TRAIN_ROOT="${TRAIN_ROOT:-/cluster/courses/cil/monocular-depth-estimation/train}"

python scripts/diagnose_boundary_errormode.py \
    --preds-dir "$PREDS_DIR" \
    --train-root "$TRAIN_ROOT" \
    "$@"
