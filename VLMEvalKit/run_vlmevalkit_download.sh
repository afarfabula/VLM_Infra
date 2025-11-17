#!/bin/bash

# 激活dataset环境
echo "激活dataset环境..."
source /data/miniforge3/etc/profile.d/conda.sh
conda activate dataset

# 检查是否成功激活环境
if [ $? -ne 0 ]; then
    echo "错误: 无法激活dataset环境"
    exit 1
fi

echo "环境已激活，开始使用VLMEvalKit下载MME数据集..."

# 运行VLMEvalKit下载脚本
python download_mme_benchmark.py --dataset mme_cot

# 检查下载结果
if [ $? -eq 0 ]; then
    echo "MME数据集下载完成!"
else
    echo "MME数据集下载失败!"
fi

echo "任务执行完毕"