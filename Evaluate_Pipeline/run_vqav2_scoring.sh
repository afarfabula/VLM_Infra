#!/bin/bash

# 简化版VQAv2推理和评分一体化脚本
# 使用集成的Python管道，所有流程都在main函数中

# 激活conda环境
if command -v conda &> /dev/null; then
    # 使用source避免'conda init'错误
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate llava
    echo "已激活conda环境: llava"
else
    echo "警告: 未找到conda命令，可能无法正确加载依赖"
fi

# 设置环境变量
export HF_HOME=/data/model/Inference_VLM/.cache
export HUGGINGFACE_HUB_CACHE=/data/model/Inference_VLM/.cache
export TRANSFORMERS_CACHE=/data/model/Inference_VLM/.cache

# 默认参数
MODEL_NAME="${1:-LLaVA-1.5-7B}"
USE_VISIONZIP="${2:-true}"  # 默认使用VisionZip
GPU_ID="${3:-1}"  # 默认使用GPU 1
LOAD_PRECISION="${4:-fp16}"  # 默认使用fp16精度
NUM_SAMPLES="${5:-200}"  # 默认推理样本数
BATCH_SIZE="${6:-32}"  # 默认批次大小
DOMINANT_COUNT="${7:-54}"  # 默认dominant token数
CONTEXTUAL_COUNT="${8:-10}"  # 默认contextual token数

# 输出目录设置
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_DIR="./results/vqav2_${TIMESTAMP}"

# 显示使用信息
echo "VQAv2推理和评分一体化脚本 (简化版)"
echo "==============================="
echo "模型: $MODEL_NAME"
echo "使用VisionZip: $USE_VISIONZIP"
echo "GPU ID: $GPU_ID"
echo "加载精度: $LOAD_PRECISION"
echo "样本数量: $NUM_SAMPLES"
echo "批次大小: $BATCH_SIZE"
echo "Dominant Token数: $DOMINANT_COUNT"
echo "Contextual Token数: $CONTEXTUAL_COUNT"
echo "结果目录: $RESULT_DIR"
echo ""

# 设置PYTHONPATH以确保能正确导入模块
export PYTHONPATH="/data/model/Inference_VLM/VLM_Infra/LLava:$PYTHONPATH"

# 运行集成的VQAv2管道
echo "=== 启动VQAv2推理和评分管道 ==="

# 根据USE_VISIONZIP的值决定是否添加--visionzip参数
VISIONZIP_FLAG=""
if [ "$USE_VISIONZIP" = "true" ]; then
    VISIONZIP_FLAG="--visionzip"
fi

CUDA_VISIBLE_DEVICES=$GPU_ID python integrated_vqav2_pipeline.py \
    --model "$MODEL_NAME" \
    $VISIONZIP_FLAG \
    --output "$RESULT_DIR" \
    --num_samples "$NUM_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --load_precision "$LOAD_PRECISION" \
    --dominant "$DOMINANT_COUNT" \
    --contextual "$CONTEXTUAL_COUNT"

# 检查执行是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "推理和评分流程完成!"
    echo "结果保存在: $RESULT_DIR"
else
    echo ""
    echo "推理和评分过程中出现错误!"
    exit 1
fi