#!/bin/bash

# 通用数据集评估启动脚本
# 支持vqav2和scienceqa数据集

# 启用错误追踪和调试模式
set -euo pipefail  # 在遇到错误时立即退出，未定义变量时报错，管道中任何命令失败都报错
exec 2>&1  # 将标准错误重定向到标准输出

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

# 模型和数据集配置
MODEL_NAME="${1:-LLaVA-1.5-7B}"
DATASET="${2:-scienceqa}"  # 默认使用scienceqa数据集
USE_VISIONZIP="${3:---visionzip}"  # 默认使用VisionZip
GPU_ID="${4:-0}"  # 默认使用GPU 1
LOAD_PRECISION="${5:-fp16}"
USE_FLASH_ATTN="${6:---use-flash-attn}"  # 默认启用Flash Attention

# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 创建基于时间戳、数据集和模型的输出目录
OUTPUT_DIR="./outputs/${TIMESTAMP}_${DATASET}_${MODEL_NAME}"
mkdir -p "$OUTPUT_DIR"

# 根据数据集选择配置文件
if [ "$DATASET" = "scienceqa" ]; then
    CONFIG_FILE="configs/scienceqa_config.json"
    
else
    CONFIG_FILE="configs/vqav2_config.json"
fi

# 通用评估
case "$DATASET" in
    "vqav2")
        SAMPLE_COUNT=50
        ;;  
    "scienceqa")
        SAMPLE_COUNT=1000
        ;;  
    *)
        SAMPLE_COUNT=50
        echo "警告: 未知数据集 $DATASET，设置样本数为默认值"
        ;;
esac

# 输出评估信息
echo "启动通用数据集评估..."
echo "模型: $MODEL_NAME"
echo "数据集: $DATASET"
echo "工作目录: $WORK_DIR"
echo "输出目录: $OUTPUT_DIR"

# 检查GPU状态
echo "=== GPU状态检查 ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

# 启动通用评估
echo "=== 启动评估 ==="
echo "数据集: $DATASET"
echo "样本数量: $SAMPLE_COUNT, Batch Size: 32"
echo "使用GPU: $GPU_ID"
echo "模型加载精度: $LOAD_PRECISION"
echo "Flash Attention: ${USE_FLASH_ATTN:+启用}${USE_FLASH_ATTN:---no-flash-attn}"
echo "配置文件: $CONFIG_FILE"

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
    --config "$CONFIG_FILE" \
    --model "$MODEL_NAME" \
    --dataset "$DATASET" \
    $USE_VISIONZIP \
    $USE_FLASH_ATTN \
    --output "$OUTPUT_DIR" \
    --num_samples $SAMPLE_COUNT \
    --batch_size 32 \
    --load_precision "$LOAD_PRECISION"

# 检查Python命令的退出状态
if [ $? -ne 0 ]; then
    echo "错误: Python评估脚本执行失败"
    exit 1
fi

echo "通用数据集评估完成!"
