# 优化版评估管道

本目录包含使用VLLM和FlashAttention优化的评估管道，旨在提高视觉语言模型在VQAv2数据集上的推理吞吐量。

## 功能特性

1. **VLLM加速**: 使用VLLM推理引擎提高批处理效率和内存利用率
2. **多GPU并行**: 支持张量并行，在多个GPU上分布模型计算
3. **FlashAttention优化**: 集成FlashAttention以提高注意力计算效率
4. **智能回退**: 当VLLM不可用时自动切换到简化版推理引擎
5. **分布式支持**: 支持多进程分布式推理

## 目录结构

```
optimized_inference/
├── __init__.py                 # 包初始化文件
├── optimized_pipeline.py       # 优化版评估管道主类
├── vllm_inference.py           # VLLM推理引擎实现
├── simple_vllm_fallback.py     # 简化版推理引擎（后备方案）
└── README.md                   # 本说明文件
```

## 安装依赖

在使用优化版评估管道之前，请确保安装以下依赖项：

```bash
pip install torch torchvision
pip install transformers
pip install accelerate
```

### 可选依赖（推荐）

为了获得最佳性能，建议安装VLLM：

```bash
pip install vllm
```

注意：VLLM安装可能需要较长时间，因为它需要编译一些CUDA扩展。如果安装失败或耗时过长，优化管道会自动降级到标准推理模式。

```bash
# 安装FlashAttention（可选）
pip install flash-attn --no-build-isolation
```

## 使用方法

### 1. 使用运行脚本

```bash
# 使用默认参数运行（4个GPU并行）
./run_optimized.sh

# 指定参数运行
./run_optimized.sh LLaVA-1.5-7B "0,1,2,3" fp16 1000 64 ./optimized_outputs
```

### 2. 直接运行Python模块

```bash
python -m optimized_inference.optimized_pipeline \
    --config configs/optimized_vqav2_config.json \
    --model LLaVA-1.5-7B \
    --output ./outputs \
    --num_samples 1000 \
    --batch_size 64 \
    --load_precision fp16 \
    --tensor_parallel_size 4 \
    --max_new_tokens 1024
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --config | configs/vqav2_config.json | 配置文件路径 |
| --model | LLaVA-1.5-7B | 模型名称 |
| --output | ./outputs | 输出目录 |
| --num_samples | 100 | 评估样本数 |
| --batch_size | 32 | 批次大小 |
| --load_precision | fp16 | 模型加载精度 |
| --tensor_parallel_size | 1 | 张量并行大小 |
| --max_new_tokens | 1024 | 最大生成token数 |

## 性能优化建议

1. **多GPU并行**: 使用`--tensor_parallel_size`参数指定GPU数量
2. **批处理大小**: 根据GPU内存调整`--batch_size`参数
3. **精度设置**: 使用fp16或bf16以提高性能
4. **内存利用率**: 通过配置文件调整GPU内存使用率

## 注意事项

1. 确保有足够的GPU内存来支持模型和批处理
2. VLLM需要特定版本的CUDA和PyTorch
3. 如果VLLM不可用，系统会自动使用简化版推理引擎
4. 多GPU并行需要正确设置CUDA_VISIBLE_DEVICES环境变量

## 故障排除

1. **VLLM导入失败**: 确保已正确安装VLLM包
2. **CUDA内存不足**: 减小批处理大小或张量并行大小
3. **模型加载失败**: 检查模型路径和权限
4. **分布式问题**: 检查环境变量RANK、WORLD_SIZE等是否正确设置