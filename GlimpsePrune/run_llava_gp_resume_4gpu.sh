#!/usr/bin/env bash
set -euo pipefail

# Change to the GlimpsePrune directory (this script lives here)
cd "$(dirname "$0")"

# Use local writable HF caches to avoid permission issues
export HF_HOME="/data/model/Inference_VLM/VLM_Infra/GlimpsePrune/datas/.cache/huggingface"
export HF_DATASETS_CACHE="/data/model/Inference_VLM/VLM_Infra/GlimpsePrune/datas/.cache/huggingface/datasets"
export WANDB_DISABLED="true"
export HF_HUB_OFFLINE="1"

# GPUs (adjust if needed)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# Defaults (override via env or CLI args)
CONFIG_DEFAULT="train_configs/llava1_5_7b_gp/llava1_5_7b_gp.yaml"
CKPT_DEFAULT="output/llava1_5_7b_gp_0801/checkpoint-600"
PORT=${PORT:-23456}
NUM_PROCS=${NUM_PROCS:-4}

# CLI args: --config PATH --ckpt PATH
CONFIG="${1:-$CONFIG_DEFAULT}"
CKPT_DIR="${2:-$CKPT_DEFAULT}"

echo "[INFO] Using GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "[INFO] Processes: ${NUM_PROCS}, Port: ${PORT}"
echo "[INFO] Config: ${CONFIG}"
echo "[INFO] Resume checkpoint: ${CKPT_DIR}"

if [ ! -d "$CKPT_DIR" ]; then
  echo "[ERROR] Checkpoint directory not found: $CKPT_DIR"
  exit 1
fi

accelerate launch \
  --num_processes "$NUM_PROCS" \
  --main_process_port "$PORT" \
  train_llava_gp.py \
  --config "$CONFIG" \
  --resume_from_checkpoint "$CKPT_DIR" \
  --load_new_modules "$CKPT_DIR"
