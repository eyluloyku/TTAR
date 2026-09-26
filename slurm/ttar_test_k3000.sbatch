#!/bin/bash
#SBATCH --job-name=ttar_test_k3000
#SBATCH --account=cil_jobs
#SBATCH --gpus=5060ti:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#
# Two-step job:
#   1. Generate Stage A test predictions (GPU)
#   2. Run TTAR K=3000 with Depth Pro anchors (CPU, random anchors, uniform smoothness)
#   3. Write submission CSV
#
# Submit:
#   sbatch slurm/ttar_test_k3000.sbatch
#
# Output:
#   submissions/ttar_dp_k3000/preds/      refined .npy per image
#   submissions/ttar_dp_k3000/submission.csv

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
echo "[slurm] GPU: $GPU_NAME  env: $CIL_ENV"

conda activate "/cluster/courses/cil/envs/envs/${CIL_ENV}"

export PYTHONUSERBASE="${HOME}/.local"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

STAGE_A_CKPT="checkpoints/depth_anything_v2_best.pt"
STAGE_A_TEST_DIR="submissions/stage_a_test/preds"
DEPTH_PRO_TEST_DIR="submissions/depth_pro/preds"
OUT_DIR="submissions/ttar_dp_k3000"

# ── step 1: Stage A test predictions ─────────────────────────────────────────
echo ""
echo "=== step 1: Stage A test predictions ==="
mkdir -p "$STAGE_A_TEST_DIR"

python -m src.finetune.predict \
    --config configs/finetune/da_v2_vits_lora.yaml \
    --checkpoint "$STAGE_A_CKPT" \
    --test-root /cluster/courses/cil/monocular-depth-estimation/test \
    --out-dir "$STAGE_A_TEST_DIR"

echo "Stage A test predictions done: $(ls $STAGE_A_TEST_DIR | wc -l) files"

# ── step 2: TTAR K=3000 with Depth Pro anchors ───────────────────────────────
echo ""
echo "=== step 2: TTAR K=3000 (rand_unif config) ==="

python scripts/ttar_test_submission.py \
    --d0-dir     "$STAGE_A_TEST_DIR" \
    --anchor-dir "$DEPTH_PRO_TEST_DIR" \
    --out-dir    "$OUT_DIR/preds" \
    --K 3000 \
    --rand-anchors \
    --alpha-smooth 0.0 \
    --lam-s 0.1

echo ""
echo "=== done ==="
echo "submission CSV: $OUT_DIR/submission.csv"
echo "Upload $OUT_DIR/submission.csv to Kaggle"
