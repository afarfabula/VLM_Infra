#!/bin/bash

# VQAv2分布式推理启动脚本
# 针对4卡4090机器优化

# 设置环境变量
export HF_HOME=/data/model/Inference_VLM/.cache
export HUGGINGFACE_HUB_CACHE=/data/model/Inference_VLM/.cache
export TRANSFORMERS_CACHE=/data/model/Inference_VLM/.cache

# 工作目录
WORK_DIR="/data/model/Inference_VLM/VLM_Infra/Evaluate_Pipeline"
cd "$WORK_DIR"

# 模型配置
MODEL_NAME="${1:-LLaVA-1.5-7B}"
USE_VISIONZIP="${2:---use-visionzip}"  # 默认使用VisionZip

# 分布式配置 - 4卡4090机器
# 方案1: 4个进程，每个进程使用1张GPU (数据并行)
# 方案2: 2个进程，每个进程使用2张GPU (模型并行)
# 这里使用方案1，更稳定且易于实现

NUM_PROCESSES=4  # 使用4个进程，对应4张4090

# 使用torchrun启动分布式推理
echo "启动VQAv2分布式评估..."
echo "模型: $MODEL_NAME"
echo "进程数: $NUM_PROCESSES"
echo "工作目录: $WORK_DIR"

# 检查GPU状态
echo "=== GPU状态检查 ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

# 启动分布式推理
echo "=== 启动分布式推理 ==="
torchrun --nproc-per-node=$NUM_PROCESSES \
    --nnodes=1 \
    --node-rank=0 \
    --master-addr=localhost \
    --master-port=29500 \
    main.py \
    --model "$MODEL_NAME" \
    $USE_VISIONZIP \
    --work-dir "./distributed_outputs" \
    --dominant \
    --contextual

echo "分布式推理完成!"