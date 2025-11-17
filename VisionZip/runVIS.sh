#!/usr/bin/env bash
# 文件路径: /data/model/Inference_VLM/VLM_Infra/VisionZip/run_llava.sh
set -euo pipefail

# 1. 初始化 Conda（根据你的安装位置自行调整）
#    下面路径是 Anaconda 默认位置，miniconda 换成 ~/miniconda3 即可
source ~/anaconda3/etc/profile.d/conda.sh

# 2. 激活环境 + 切目录
conda activate LLava
cd /data/model/Inference_VLM/VLM_Infra/VisionZip

# 3. 导出缓存变量
export HF_HOME=/data/model/Inference_VLM/.cache
export HUGGINGFACE_HUB_CACHE=/data/model/Inference_VLM/.cache
export TRANSFORMERS_CACHE=/data/model/Inference_VLM/.cache

# 4. 跑
python visionzip_cli.py \
  --model-path /data/model/Inference_VLM/models-LLava-1.5-7B \
  --image-file /data/model/Inference_VLM/sample_dog.png \
  --dominant 54\
  --contextual 10\
  --load-4bit \
  --max-new-tokens 512