#!/bin/bash

# VQAv2批量并行推理启动脚本
# 专注于显存高效的批量并行推理

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

# 显存优化参数
MAX_BATCH_SIZE="${6:-16}"  # 最大批量大小（根据显存调整）
NUM_WORKERS="${7:-4}"  # 并行工作线程数（控制显存使用）

# 批处理并行推理
echo "启动VQAv2批量并行评估..."
echo "模型: $MODEL_NAME"
echo "工作目录: $WORK_DIR"
echo "显存优化参数:"
echo "  最大批量大小: $MAX_BATCH_SIZE"
echo "  并行工作线程数: $NUM_WORKERS"

# 检查GPU状态
echo "=== GPU状态检查 ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

# 启动批量并行推理
echo "=== 启动批量并行推理 ==="
echo "样本数量: 1000, Batch Size: 32"
echo "使用GPU: $GPU_ID"
echo "模型加载精度: $LOAD_PRECISION"
echo "Flash Attention: ${USE_FLASH_ATTN:+启用}${USE_FLASH_ATTN:---no-flash-attn}"
echo "显存优化: 最大批量大小=$MAX_BATCH_SIZE, 工作线程数=$NUM_WORKERS"

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
    --config "configs/vqav2_config.json" \
    --model "$MODEL_NAME" \
    $USE_VISIONZIP \
    $USE_FLASH_ATTN \
    --output "./optimized_outputs" \
    --num_samples 1000 \
    --batch_size 32 \
    --load_precision "$LOAD_PRECISION" \
    --num_workers "$NUM_WORKERS" \
    --max_batch_size "$MAX_BATCH_SIZE"

echo "批量并行推理完成!"
echo "注意：可通过调整脚本参数优化显存使用："
echo "  1. 减小NUM_WORKERS参数（当前：$NUM_WORKERS）以降低并行度"
echo "  2. 减小MAX_BATCH_SIZE参数（当前：$MAX_BATCH_SIZE）以降低单次处理的样本数"
echo "  3. 使用8bit或4bit量化以减少模型本身的显存占用"
