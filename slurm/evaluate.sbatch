#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --account=cil_jobs
#SBATCH --gpus=5060ti:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
# Pass --job-name=eval_<run> to discriminate: logs/slurm/eval_vits_lora_r8_v1_53412.out
#
# Evaluate a model on the held-out validation split.
#
#   # Zero-shot baseline (pretrained DA-V2, no LoRA):
#   sbatch slurm/evaluate.sbatch configs/finetune/da_v2_vits_lora.yaml --no-lora
#
#   # A trained checkpoint:
#   sbatch slurm/evaluate.sbatch configs/finetune/da_v2_vits_lora.yaml \
#       --checkpoint outputs/stage_a/vits_lora_r8/checkpoints/best.pt

set -euo pipefail

. /etc/profile.d/modules.sh

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[slurm] GPU: $GPU_NAME"

if [[ "${GPU_NAME}" == *"1080"* ]]; then
    CIL_ENV="${CIL_ENV:-monocular-depth-estimation}"
else
    CIL_ENV="${CIL_ENV:-monodepth-5060}"
fi
echo "[slurm] conda env: $CIL_ENV"

module load cuda/12.6.0
source /cluster/courses/cil/envs/etc/profile.d/conda.sh
conda activate "/cluster/courses/cil/envs/envs/${CIL_ENV}"

export PYTHONUSERBASE="${HOME}/.local"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

CONFIG="${1:?Usage: sbatch evaluate.sbatch <config.yaml> [--checkpoint ... | --no-lora]}"
shift

python -m src.finetune.evaluate --config "$CONFIG" "$@"
