#!/bin/bash
#SBATCH --job-name=tail10
#SBATCH --account=cil_jobs
#SBATCH --gpus=5060ti:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#
# Tail-10% deep-dive analysis. CPU only; GPU requested to hit the 5060-ti env.

set -euo pipefail
. /etc/profile.d/modules.sh
module load cuda/12.6.0
source /cluster/courses/cil/envs/etc/profile.d/conda.sh
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
if [[ "${GPU_NAME}" == *"1080"* ]]; then CIL_ENV="${CIL_ENV:-monocular-depth-estimation}"
else CIL_ENV="${CIL_ENV:-monodepth-5060}"; fi
conda activate "/cluster/courses/cil/envs/envs/${CIL_ENV}"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
python scripts/analyze_tail10.py "$@"
