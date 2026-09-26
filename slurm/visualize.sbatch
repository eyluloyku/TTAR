#!/bin/bash
#SBATCH --job-name=viz
#SBATCH --account=cil_jobs
#SBATCH --gpus=5060ti:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#
# Render 2x2 RGB/GT/Pred/error panels for a set of image names.
#
#   sbatch slurm/visualize.sbatch configs/finetune/da_v2_vits_lora.yaml \
#       outputs/inspections/zero_shot_worst \
#       --no-lora \
#       --names train_012119_rgb.png train_002809_rgb.png ...

set -euo pipefail

. /etc/profile.d/modules.sh
module load cuda/12.6.0

source /cluster/courses/cil/envs/etc/profile.d/conda.sh
conda activate /cluster/courses/cil/envs/envs/monodepth-5060

export PYTHONUSERBASE="${HOME}/.local"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

CONFIG="${1:?Usage: sbatch visualize.sbatch <config.yaml> <out_dir> [--no-lora|--checkpoint PATH] --names ...}"
OUT_DIR="${2:?missing out_dir}"
shift 2

python scripts/visualize_cases.py --config "$CONFIG" --out-dir "$OUT_DIR" "$@"
