#!/usr/bin/env bash
set -euo pipefail

# Change to the GlimpsePrune directory (this script lives here)
cd "$(dirname "$0")"

# Use local writable HF caches to avoid permission issues
export HF_HOME="/data/model/Inference_VLM/VLM_Infra/GlimpsePrune/datas/.cache/huggingface"
export HF_DATASETS_CACHE="/data/model/Inference_VLM/VLM_Infra/GlimpsePrune/datas/.cache/huggingface/datasets"
export WANDB_DISABLED="true"
export HF_HUB_OFFLINE="1"

# GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch \
  --num_processes 4 \
  --main_process_port 23456 \
  train_llava_gp.py \
  --config train_configs/llava1_5_7b_gp/llava1_5_7b_gp.yaml

