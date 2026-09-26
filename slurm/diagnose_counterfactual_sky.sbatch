#!/bin/bash
#SBATCH --job-name=cf_sky
#SBATCH --account=cil_jobs
#SBATCH --gpus=5060ti:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#
# Stage B v5 — counterfactual sky-ablation headroom probe.
# Re-runs the Stage A model on sky-inpainted images (GPU).
#
#   sbatch slurm/diagnose_counterfactual_sky.sbatch \
#       --out-dir outputs/diagnostics/counterfactual_sky/ --limit 60

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

python scripts/diagnose_counterfactual_sky.py \
    --config configs/finetune/da_v2_vits_lora.yaml \
    --checkpoint outputs/stage_a/vits_lora_r8/checkpoints/best.pt \
    --train-root /cluster/courses/cil/monocular-depth-estimation/train \
    --preds-dir outputs/val_preds/vits_lora_r8/ \
    "$@"
