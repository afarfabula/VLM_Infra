#!/bin/bash

#!/bin/bash

# ScienceQA单进程推理启动脚本
# 用于测试和调试

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
LOAD_PRECISION="${4:-fp16}"
USE_FLASH_ATTN="${5:---use-flash-attn}"  # 默认启用Flash Attention 

# 单进程推理
>&2 echo "启动ScienceQA单进程评估..."
>&2 echo "模型: $MODEL_NAME"
>&2 echo "工作目录: $WORK_DIR"

# 检查GPU状态
>&2 echo "=== GPU状态检查 ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

# 启动单进程推理
>&2 echo "=== 启动单进程推理 ==="
>&2 echo "样本数量: 100, Batch Size: 32"
>&2 echo "使用GPU: $GPU_ID"
>&2 echo "模型加载精度: $LOAD_PRECISION"
>&2 echo "Flash Attention: ${USE_FLASH_ATTN:+启用}${USE_FLASH_ATTN:---no-flash-attn}"
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
    --config "configs/scienceqa_config.json" \
    --dataset "scienceqa" \
    --model "$MODEL_NAME" \
    $USE_VISIONZIP \
    $USE_FLASH_ATTN \
    --output "./single_outputs" \
    --num_samples 100 \
    --batch_size 32 \
    --load_precision "$LOAD_PRECISION"

>&2 echo "单进程推理完成!"