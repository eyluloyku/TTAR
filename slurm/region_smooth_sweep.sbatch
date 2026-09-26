#!/bin/bash
#SBATCH --job-name=region_smooth_sweep
#SBATCH --account=cil_jobs
#SBATCH --gpus=5060ti:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#
# Sweep alpha for the SAM-region depth smoothing refiner on the val split.
#
# The first alpha pays the SAM extraction cost (~30 min on 1159 images).
# All subsequent alphas reuse the cached masks under
# outputs/stage_b_cache/sam_regions/, so they finish in ~1-2 min each.
#
# Usage:
#   sbatch slurm/region_smooth_sweep.sbatch
#
# Overrides via env vars:
#   PREDS_DIR            Stage A predictions dir (default: outputs/val_preds/vits_lora_r8/)
#   TRAIN_ROOT           train data root (default: cluster train split)
#   ALPHAS               space-separated values (default: "0.2 0.4 0.6 0.8")
#   LIMIT                cap on number of val images (default: 200; "" = full set)
#   AGGREGATOR           "median" or "plane" (default: "median")
#   MAX_PLANE_RESIDUAL   gate threshold for "plane": regions whose log-depth
#                        residual exceeds this are left unsmoothed. Unset =
#                        no gate.
#   BOUNDARY_ONLY        "true" or "false" (default: false). When true, the
#                        per-region target is applied only to a BAND_WIDTH-pixel
#                        rim around each region — corrects DA-V2's blurry
#                        boundary transitions while leaving region interiors
#                        untouched.
#   BAND_WIDTH           rim width in pixels when BOUNDARY_ONLY=true (default: 3).
#
# The default 200-image subsample is enough to find the alpha curve while
# staying within the 3h budget at the full SAM-quality config used by B1.
#
# Examples:
#   ALPHAS="0.3 0.5 0.7" sbatch slurm/region_smooth_sweep.sbatch
#   LIMIT="" sbatch slurm/region_smooth_sweep.sbatch              # full val
#   AGGREGATOR=plane sbatch slurm/region_smooth_sweep.sbatch       # plane fit
#   AGGREGATOR=plane MAX_PLANE_RESIDUAL=0.2 sbatch slurm/region_smooth_sweep.sbatch
#   AGGREGATOR=plane BOUNDARY_ONLY=true BAND_WIDTH=3 sbatch slurm/region_smooth_sweep.sbatch

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
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

PREDS_DIR="${PREDS_DIR:-outputs/val_preds/vits_lora_r8/}"
TRAIN_ROOT="${TRAIN_ROOT:-/cluster/courses/cil/monocular-depth-estimation/train}"
ALPHAS="${ALPHAS:-0.2 0.4 0.6 0.8}"
LIMIT="${LIMIT-200}"   # use LIMIT="" to disable the subsample cap
AGGREGATOR="${AGGREGATOR:-median}"

LIMIT_ARG=()
if [[ -n "$LIMIT" ]]; then
    LIMIT_ARG=(--limit "$LIMIT")
fi

EXTRA_REFINER_KWARGS=()
if [[ -n "${MAX_PLANE_RESIDUAL:-}" ]]; then
    EXTRA_REFINER_KWARGS+=("max_plane_residual=$MAX_PLANE_RESIDUAL")
fi
if [[ -n "${BOUNDARY_ONLY:-}" ]]; then
    EXTRA_REFINER_KWARGS+=("boundary_only=$BOUNDARY_ONLY")
fi
if [[ -n "${BAND_WIDTH:-}" ]]; then
    EXTRA_REFINER_KWARGS+=("band_width=$BAND_WIDTH")
fi

echo "[sweep] preds-dir:           $PREDS_DIR"
echo "[sweep] train-root:          $TRAIN_ROOT"
echo "[sweep] alphas:              $ALPHAS"
echo "[sweep] limit:               ${LIMIT:-<full val>}"
echo "[sweep] aggregator:          $AGGREGATOR"
echo "[sweep] max_plane_residual:  ${MAX_PLANE_RESIDUAL:-<no gate>}"
echo "[sweep] boundary_only:       ${BOUNDARY_ONLY:-false}"
echo "[sweep] band_width:          ${BAND_WIDTH:-3 (default)}"

for a in $ALPHAS; do
    echo ""
    echo "============================================================"
    echo "=== alpha=$a   agg=$AGGREGATOR   gate=${MAX_PLANE_RESIDUAL:-inf}   boundary=${BOUNDARY_ONLY:-false}/${BAND_WIDTH:-3}"
    echo "============================================================"
    python -u scripts/eval_refine.py \
        --preds-dir "$PREDS_DIR" \
        --train-root "$TRAIN_ROOT" \
        --refiner region_smooth \
        --refiner-kwargs "alpha=$a" "aggregator=$AGGREGATOR" "${EXTRA_REFINER_KWARGS[@]}" \
        "${LIMIT_ARG[@]}"
done

echo ""
echo "[sweep] done"
