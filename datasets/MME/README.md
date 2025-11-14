# MME (Multimodal Model Evaluation) 数据集

MME (Multimodal Model Evaluation) 是一个全面的多模态模型评估基准，包含感知和认知两个维度的评估任务。

## 数据集概述

MME数据集包含两个主要版本：

1. **MME_CoT** - 思维链推理版本
2. **MMEReasoning** - 推理版本

## 下载方法

### 方法一：使用VLMEvalKit下载脚本（推荐）

```bash
# 进入VLMEvalKit目录
cd /data/model/Inference_VLM/VLM_Infra/VLMEvalKit

# 下载所有MME数据集
python download_mme_benchmark.py --dataset all

# 仅下载MME_CoT数据集
python download_mme_benchmark.py --dataset mme_cot

# 仅下载MMEReasoning数据集
python download_mme_benchmark.py --dataset mme_reasoning

# 使用ModelScope下载（如果HuggingFace访问受限）
python download_mme_benchmark.py --dataset all --use-modelscope

# 指定自定义缓存目录
python download_mme_benchmark.py --dataset all --cache-dir /path/to/cache
```

### 方法二：使用原始下载脚本

```bash
# 进入MME数据集目录
cd /data/model/Inference_VLM/VLM_Infra/datasets/MME

# 运行下载脚本
python download_mme.py
```

## 测试下载功能

```bash
# 测试VLMEvalKit中的MME数据集功能
cd /data/model/Inference_VLM/VLM_Infra/VLMEvalKit
python test_mme_download.py
```

## 数据集结构

下载完成后，数据集将存储在以下位置：

```
/data/model/Inference_VLM/VLM_Infra/datasets/MME/
├── images/           # 图像文件
├── annotations/      # 标注文件
├── mme.tsv          # 数据集索引文件
├── download_mme.py  # 下载脚本
└── README.md        # 说明文档
```

## 在代码中使用

### 使用VLMEvalKit加载数据集

```python
from vlmeval.dataset import build_dataset

# 加载MME_CoT数据集
mme_cot_dataset = build_dataset('MME_CoT')

# 加载MMEReasoning数据集
mme_reasoning_dataset = build_dataset('MMEReasoning')

# 访问数据
for item in mme_cot_dataset.data:
    image_path = item['image']
    question = item['question']
    # 处理数据...
```

### 直接使用数据集文件

```python
import pandas as pd

# 读取TSV文件
df = pd.read_csv('/data/model/Inference_VLM/VLM_Infra/datasets/MME/mme.tsv', sep='\t')

# 处理数据
for index, row in df.iterrows():
    image_path = row['image']
    question = row['question']
    # 处理数据...
```

## 注意事项

1. **网络要求**：下载需要访问HuggingFace或ModelSpace
2. **存储空间**：完整数据集需要约1-2GB存储空间
3. **缓存管理**：数据集会自动缓存，可通过环境变量控制缓存位置
4. **版本兼容**：确保VLMEvalKit版本支持MME数据集

## 故障排除

### 下载失败
- 检查网络连接
- 尝试使用`--use-modelscope`选项
- 检查磁盘空间

### 导入错误
- 确保VLMEvalKit已正确安装
- 检查Python环境
- 验证依赖包版本

## 相关文件

- `download_mme_benchmark.py` - VLMEvalKit下载脚本
- `test_mme_download.py` - 功能测试脚本
- `download_mme.py` - 原始下载脚本

## 参考链接

- [MME官方仓库](https://github.com/BradyFU/MME)
- [VLMEvalKit文档](https://github.com/open-compass/VLMEvalKit)
- [HuggingFace数据集](https://huggingface.co/datasets/BradyFU/MME)