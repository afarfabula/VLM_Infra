#!/bin/bash

# VQAv2单进程推理启动脚本
# 用于测试和调试

# 设置环境变量
export HF_HOME=/data/model/Inference_VLM/.cache
export HUGGINGFACE_HUB_CACHE=/data/model/Inference_VLM/.cache
export TRANSFORMERS_CACHE=/data/model/Inference_VLM/.cache

# 工作目录
WORK_DIR="/data/model/Inference_VLM/VLM_Infra/Evaluate_Pipeline"
cd "$WORK_DIR"

# 模型配置
MODEL_NAME="${1:-LLaVA-1.5-7B}"
USE_VISIONZIP="${2:---visionzip}"  # 默认使用VisionZip
GPU_ID="${3:-1}"  # 默认使用GPU 1
#LOAD_PRECISION="${4:-8bit}"  # 默认使用8bit量化加载
LOAD_PRECISION="${4:-fp16}" 

# 单进程推理
echo "启动VQAv2单进程评估..."
echo "模型: $MODEL_NAME"
echo "工作目录: $WORK_DIR"

# 检查GPU状态
echo "=== GPU状态检查 ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

# 启动单进程推理
echo "=== 启动单进程推理 ==="
echo "样本数量: 100, Batch Size: 32"
echo "使用GPU: $GPU_ID"
echo "模型加载精度: $LOAD_PRECISION"
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
    --config "configs/vqav2_config.json" \
    --model "$MODEL_NAME" \
    $USE_VISIONZIP \
    --output "./single_outputs" \
    --num_samples 100 \
    --batch_size 32 \
    --load_precision "$LOAD_PRECISION"

echo "单进程推理完成!"