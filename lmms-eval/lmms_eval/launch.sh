#!/usr/bin/env bash
set -euo pipefail

# 激活 lmms-eval Conda 环境（尝试常见安装路径）
# 尝试加载conda.sh，即使conda命令已经可用，也需要它来使用conda activate
conda_sh_found=false
for cand in \
  "$HOME/miniconda3/etc/profile.d/conda.sh" \
  "$HOME/miniforge3/etc/profile.d/conda.sh" \
  "/data/miniforge3/etc/profile.d/conda.sh" \
  "/opt/conda/etc/profile.d/conda.sh"
do
  if [ -f "$cand" ]; then
    # shellcheck disable=SC1090
    source "$cand"
    conda_sh_found=true
    break
  fi
done

# 如果没有找到conda.sh，检查conda命令是否可用
if ! command -v conda >/dev/null 2>&1; then
  echo "Conda 未找到，请先安装或加载 Conda 后再运行本脚本"
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda 未找到，请先安装或加载 Conda 后再运行本脚本"
  exit 1
fi

conda activate lmms-eval

# 进入工程目录
cd /data/model/Inference_VLM/VLM_Infra/lmms-eval

# 缓存目录（HF 推荐使用 HF_HOME；保留 TRANSFORMERS_CACHE 兼容旧代码）
export HF_HOME="/data/model/Inference_VLM/.cache"
export HUGGINGFACE_HUB_CACHE="/data/model/Inference_VLM/.cache"
export TRANSFORMERS_CACHE="/data/model/Inference_VLM/.cache"

# 可选参数：PRECISION 默认 4bit；SUFFIX 默认 pope
PRECISION="${1:-4bit}"
SUFFIX="${2:-pope}"

python -m lmms_eval \
  --model visionzip_llava \
  --model_args pretrained=/data/model/Inference_VLM/models-LLava-1.5-7B,device=cuda,load_precision="${PRECISION}",use_flash_attn=True \
  --tasks pope \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix "${SUFFIX}" \
  --output_path ./logs/