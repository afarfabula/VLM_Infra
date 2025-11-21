#!/bin/bash

# VisionZip运行脚本
# 封装了完整的环境变量和参数配置

# 使用source来激活conda环境，避免'conda init'错误
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate llava

# 设置缓存目录环境变量
export HF_HOME="/data/model/Inference_VLM/.cache"
export HUGGINGFACE_HUB_CACHE="/data/model/Inference_VLM/.cache"
export TRANSFORMERS_CACHE="/data/model/Inference_VLM/.cache"

# 运行VisionZip CLI脚本
# 已添加--load-fp16参数支持
python /data/model/Inference_VLM/VLM_Infra/VisionZip/visionzip_cli.py \
    --model-path /data/model/Inference_VLM/models-LLava-1.5-7B \
    --image-file /data/model/Inference_VLM/VLM_Infra/VisionZip/sample_dog.png \
    --load-fp16 \
    --max-new-tokens 100 \
    --device cuda