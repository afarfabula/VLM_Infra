#!/usr/bin/env bash
set -euo pipefail
ENV=/data/miniforge3/envs/llava-next
PY=$ENV/bin/python
PIP=$ENV/bin/pip
export HF_HOME=/data/model/Inference_VLM/.cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME
mkdir -p "$HF_HOME"
if [ -n "${HF_TOKEN:-}" ]; then export HF_TOKEN; fi
cd /data/model/Inference_VLM/VLM_Infra/LLaVA-NeXT
if ! "$PY" -c "import flash_attn" >/dev/null 2>&1; then
  export CUDA_HOME=$ENV
  export PATH=$CUDA_HOME/bin:$PATH
  export TORCH_CUDA_ARCH_LIST=8.9
  export MAX_JOBS=4
  "$PIP" install --no-build-isolation --no-cache-dir flash-attn==2.6.3
fi
"$PY" - <<'PY'
import flash_attn, torch
print("flash_attn_version:", getattr(flash_attn, "__version__", "unknown"))
print("cuda_available:", torch.cuda.is_available())
PY
"$PY" test_demo.py
