#!/bin/bash

# VQAv2评分脚本
# 使用多个GT答案进行更准确的评分

# 默认参数
PRED_JSONL="${1:-./predictions.jsonl}"
ANNOTATIONS_JSON="${2:-/data/model/Inference_VLM/VLM_Infra/datasets/VQAv2/annotations/v2_mscoco_val2014_annotations.json}"
OUTPUT_PATH="${3:-../results/scoring_result.json}"

# 显示使用信息
echo "VQAv2评分脚本"
echo "============="
echo "预测文件: $PRED_JSONL"
echo "标注文件: $ANNOTATIONS_JSON"
echo "输出文件: $OUTPUT_PATH"
echo ""

# 检查文件是否存在
if [ ! -f "$PRED_JSONL" ]; then
    echo "错误: 预测文件不存在: $PRED_JSONL"
    echo "请提供有效的预测文件路径作为第一个参数"
    exit 1
fi

if [ ! -f "$ANNOTATIONS_JSON" ]; then
    echo "错误: 标注文件不存在: $ANNOTATIONS_JSON"
    echo "请确保标注文件存在于指定路径"
    exit 1
fi

# 创建输出目录
mkdir -p "$(dirname "$OUTPUT_PATH")"

# 运行评分
echo "正在运行VQAv2评分..."
python "$(dirname "$0")/score_vqav2_with_multiple_gt.py" \
    --pred-jsonl "$PRED_JSONL" \
    --annotations-json "$ANNOTATIONS_JSON" \
    --output "$OUTPUT_PATH"

# 检查评分是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "评分完成!"
    echo "详细结果已保存到: $OUTPUT_PATH"
else
    echo ""
    echo "评分过程中出现错误!"
    exit 1
fi