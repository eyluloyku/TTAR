#!/bin/bash
#SBATCH --job-name=stage_a
#SBATCH --account=cil_jobs
#SBATCH --gpus=5060ti:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
# %x = job name, %j = job id. Pass --job-name=<run-name> on the sbatch CLI
# to get log files like: logs/slurm/vits_lora_r8_v1_53410.out
#
# ETH student cluster notes (see isg.inf.ethz.ch docs):
#   - Default per-job limits: 2 CPUs, 24 GB RAM, 40 GB /tmp, 1 running + 2 queued.
#     Requests above these likely get trimmed/rejected, so we rely on defaults.
#   - `--time` is required (default is 60 min). RTX GPUs have a 7-day runtime cap.
#   - `module load` needs /etc/profile.d/modules.sh sourced first.
#   - logs/slurm/ must exist before submission (mkdir -p logs/slurm).
#
# Prerequisites (run once on the cluster — see README §0 for full setup):
#   pip install --user -r requirements.txt
#   wget -O checkpoints/depth_anything_v2_vits.pth <HF url>   # Stage A backbone
#   wget -O checkpoints/sam_vit_b.pth <fbaipublicfiles url>   # Stage B segmentation
#
# Submit from the repo root:
#   mkdir -p logs/slurm
#   sbatch slurm/train_finetune.sbatch configs/finetune/da_v2_vits_lora.yaml
# Pass additional dot-path overrides after the config path:
#   sbatch slurm/train_finetune.sbatch configs/finetune/da_v2_vits_lora.yaml \
#       train.batch_size=4 train.epochs=20
#
# GPU selection (override via sbatch CLI):
#   --gpus=5060ti:1   RTX 5060 Ti  (Blackwell) → monodepth-5060 env
#   --gpus=2080ti:1   RTX 2080 Ti               → monodepth-5060 env
#   --gpus=1080ti:1   GTX 1080 Ti               → monocular-depth-estimation env
#   --gpus=gb10:1     NVIDIA GB10 (128GB, 24h cap) → monodepth-5060 env
# The env is auto-picked from the assigned GPU name below (override with CIL_ENV=...).

set -euo pipefail

. /etc/profile.d/modules.sh

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[slurm] GPU: $GPU_NAME"

# 1080 Ti is the only card still tied to the old cu124 PyTorch env.
if [[ "${GPU_NAME}" == *"1080"* ]]; then
    CIL_ENV="${CIL_ENV:-monocular-depth-estimation}"
else
    CIL_ENV="${CIL_ENV:-monodepth-5060}"
fi
echo "[slurm] conda env: $CIL_ENV"

module load cuda/12.6.0

source /cluster/courses/cil/envs/etc/profile.d/conda.sh
conda activate "/cluster/courses/cil/envs/envs/${CIL_ENV}"

# Make user-installed (pip install --user) packages importable.
export PYTHONUSERBASE="${HOME}/.local"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

CONFIG="${1:?Usage: sbatch train_finetune.sbatch <config.yaml> [train.lr=... ...]}"
shift || true

echo "[slurm] config: $CONFIG"
echo "[slurm] overrides: $*"

if [[ $# -gt 0 ]]; then
    python -m src.finetune.train --config "$CONFIG" --set "$@"
else
    python -m src.finetune.train --config "$CONFIG"
fi
