#!/bin/bash
#SBATCH --job-name=predict
#SBATCH --account=cil_jobs
#SBATCH --gpus=5060ti:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#
# Generate test-set depth predictions for Kaggle submission.
#
# Test-set mode (Kaggle submission):
#   sbatch slurm/predict.sbatch configs/finetune/da_v2_vits_lora.yaml \
#       outputs/stage_a/vits_lora_r8/checkpoints/best.pt \
#       submissions/vits_lora_r8/preds/
#
# Val-set mode (error analysis):
#   sbatch slurm/predict.sbatch configs/finetune/da_v2_vits_lora.yaml \
#       outputs/stage_a/vits_lora_r8/checkpoints/best.pt \
#       outputs/val_preds/vits_lora_r8/ --val
#
# With guided filter:
#   sbatch slurm/predict.sbatch configs/finetune/da_v2_vits_lora.yaml \
#       checkpoints/friend_best.pt outputs/val_preds/with_gf/ --val --guided-filter

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

CONFIG="${1:?Usage: sbatch predict.sbatch <config.yaml> <checkpoint> <out_dir> [--val] [--guided-filter ...]}"
CHECKPOINT="${2:?missing checkpoint path}"
OUT_DIR="${3:?missing out_dir}"
shift 3

# Default to test-set mode if neither --val nor --test-root is provided.
if [[ "$*" != *"--val"* && "$*" != *"--test-root"* ]]; then
    set -- "--test-root" "/cluster/courses/cil/monocular-depth-estimation/test" "$@"
fi

python -m src.finetune.predict \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --out-dir "$OUT_DIR" \
    "$@"
