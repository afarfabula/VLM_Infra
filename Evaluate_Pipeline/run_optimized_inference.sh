#!/bin/bash

# VQAv2优化推理启动脚本
# 用于测试显存优化的共享模型推理实现

# 使用source来激活conda环境，避免'conda init'错误
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate llava

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
LOAD_PRECISION="${4:-fp16}"  # 默认使用fp16精度
USE_FLASH_ATTN="${5:---use-flash-attn}"  # 默认启用Flash Attention

# 优化推理参数
USE_SHARED_MODEL="--use-shared-model"  # 使用共享模型推理
NUM_WORKERS="${6:-4}"  # 预处理并行工作线程数

# 优化推理
echo "启动VQAv2优化推理评估..."
echo "模型: $MODEL_NAME"
echo "工作目录: $WORK_DIR"
echo "使用共享模型推理: 是"
echo "预处理工作线程数: $NUM_WORKERS"

# 检查GPU状态
echo "=== GPU状态检查 ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

# 启动优化推理
echo "=== 启动优化推理 ==="
echo "样本数量: 1000, Batch Size: 32"
echo "使用GPU: $GPU_ID"
echo "模型加载精度: $LOAD_PRECISION"
echo "Flash Attention: ${USE_FLASH_ATTN:+启用}${USE_FLASH_ATTN:---no-flash-attn}"
echo "显存优化: 启用 (模型权重共享+内存监控)"

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
    --config "configs/vqav2_config.json" \
    --model "$MODEL_NAME" \
    $USE_VISIONZIP \
    $USE_FLASH_ATTN \
    $USE_SHARED_MODEL \
    --num-workers "$NUM_WORKERS" \
    --output "./optimized_outputs" \
    --num_samples 1000 \
    --batch_size 32 \
    --load_precision "$LOAD_PRECISION"

echo "优化推理完成!"
