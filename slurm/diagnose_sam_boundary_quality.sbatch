#!/bin/bash
#SBATCH --job-name=bnd_phase0b
#SBATCH --account=cil_jobs
#SBATCH --gpus=5060ti:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#
# Stage B v5 — Phase 0b (Q0.2): SAM boundary quality + partition caching.
# Runs SAM on val (GPU) — partitions are cached for reuse in Phases 1-2.
#
# Full val:
#   sbatch slurm/diagnose_sam_boundary_quality.sbatch \
#       --out-dir outputs/diagnostics/boundary_phase0b/

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

python scripts/diagnose_sam_boundary_quality.py \
    --preds-dir "$PREDS_DIR" \
    --train-root "$TRAIN_ROOT" \
    --sam-checkpoint checkpoints/sam_vit_b.pth \
    --cache-dir outputs/stage_b_cache/sam_partitions/ \
    "$@"
