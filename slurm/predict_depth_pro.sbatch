#!/bin/bash
#SBATCH --job-name=depth_pro_predict
#SBATCH --account=cil_jobs
#SBATCH --gpus=5060ti:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#
# Zero-shot Depth Pro predictions on the val or test split.
#
# Val-set mode (error analysis / comparison with Stage A):
#   sbatch slurm/predict_depth_pro.sbatch \
#       checkpoints/depth_pro.pt \
#       outputs/val_preds/depth_pro/ \
#       --val --config configs/finetune/da_v2_vits_lora.yaml
#
# Test-set mode (Kaggle submission):
#   sbatch slurm/predict_depth_pro.sbatch \
#       checkpoints/depth_pro.pt \
#       submissions/depth_pro/preds/ \
#       --test-root /cluster/courses/cil/monocular-depth-estimation/test
#
# Smoke test (first 20 val images):
#   sbatch slurm/predict_depth_pro.sbatch \
#       checkpoints/depth_pro.pt \
#       outputs/val_preds/depth_pro_smoke/ \
#       --val --config configs/finetune/da_v2_vits_lora.yaml --limit 20

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

CHECKPOINT="${1:?Usage: sbatch predict_depth_pro.sbatch <checkpoint> <out_dir> [--val --config ...] [--test-root ...]}"
OUT_DIR="${2:?missing out_dir}"
shift 2

if [[ "$*" != *"--val"* && "$*" != *"--test-root"* ]]; then
    set -- "--test-root" "/cluster/courses/cil/monocular-depth-estimation/test" "$@"
fi

python -m src.depth_pro.predict \
    --checkpoint "$CHECKPOINT" \
    --out-dir "$OUT_DIR" \
    --skip-existing \
    "$@"
