#!/bin/bash
#SBATCH --job-name=tta_var
#SBATCH --account=cil_jobs
#SBATCH --gpus=5060ti:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#
# Precompute per-image TTA log-depth variance for val (or test) split.
# 5 forward passes per image (id, hflip, +bright, -bright, hflip+bright).
# Writes one <stem>.npy per image into --out-dir.
#
# Val:
#   sbatch slurm/precompute_tta_variance.sbatch configs/finetune/da_v2_vits_lora.yaml \
#       outputs/stage_a/vits_lora_r8/checkpoints/best.pt \
#       outputs/val_preds/vits_lora_r8/tta_var/ --val
#
# Test:
#   sbatch slurm/precompute_tta_variance.sbatch configs/finetune/da_v2_vits_lora.yaml \
#       outputs/stage_a/vits_lora_r8/checkpoints/best.pt \
#       outputs/test_preds/vits_lora_r8/tta_var/ \
#       --test-root /cluster/courses/cil/monocular-depth-estimation/test

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

CONFIG="${1:?Usage: sbatch precompute_tta_variance.sbatch <config.yaml> <checkpoint> <out_dir> [--val | --test-root <dir>]}"
CHECKPOINT="${2:?missing checkpoint path}"
OUT_DIR="${3:?missing out_dir}"
shift 3

python scripts/precompute_tta_variance.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --out-dir "$OUT_DIR" \
    "$@"
