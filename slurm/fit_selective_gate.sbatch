#!/bin/bash
#SBATCH --job-name=gate
#SBATCH --account=cil_jobs
#SBATCH --gpus=5060ti:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#
# Phase 3 — fit + evaluate the selective-application gate on Phase 1c results.
# Pure CPU work (sklearn LogisticRegression + cv2 for grad features); GPU
# requested only to land on the 5060-ti env where deps are installed.
#
# Example:
#   sbatch slurm/fit_selective_gate.sbatch \
#       --eval-json outputs/diagnostics/hm_oracle_v3_phase1/results.json \
#       --preds-dir outputs/val_preds/vits_lora_r8/ \
#       --tta-var-dir outputs/val_preds/vits_lora_r8/tta_var/ \
#       --out-dir outputs/diagnostics/gate_v3/

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

python scripts/fit_selective_gate.py "$@"
