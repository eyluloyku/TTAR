#!/bin/bash
#SBATCH --job-name=depthlm_sanity
#SBATCH --account=cil_jobs
#SBATCH --gpus=5060ti:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#
# DepthLM backend sanity test — load the model, query a handful of anchors
# on one val image, print parsed depths.
#
#   sbatch slurm/sanity_depthlm.sbatch --stem train_007817

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

python scripts/sanity_depthlm.py \
    --preds-dir outputs/val_preds/vits_lora_r8/ \
    --train-root /cluster/courses/cil/monocular-depth-estimation/train \
    "$@"
