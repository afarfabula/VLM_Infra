# VQAv2 多答案评分系统

## 简介

本系统实现了VQAv2数据集的官方软准确率评分算法，支持使用多个标准答案进行更准确的模型评估。

## 特性

- 支持VQAv2官方软准确率计算方法
- 使用多个标准答案进行评分，提高评估准确性
- 包含文本规范化和答案提取功能
- 提供详细的评分报告（按问题类型和答案类型分类）

## 文件说明

- `score_vqav2_with_multiple_gt.py`: 主评分脚本
- `run_vqav2_scoring.sh`: 评分执行脚本
- `sample_predictions.jsonl`: 示例预测文件

## 使用方法

### 1. 直接运行Python脚本

```bash
python score_vqav2_with_multiple_gt.py \
    --pred-jsonl /path/to/predictions.jsonl \
    --annotations-json /path/to/v2_mscoco_val2014_annotations.json \
    --output /path/to/scoring_result.json
```

### 2. 使用执行脚本

```bash
./run_vqav2_scoring.sh \
    /path/to/predictions.jsonl \
    /path/to/v2_mscoco_val2014_annotations.json \
    /path/to/scoring_result.json
```

## 预测文件格式

预测文件应为JSONL格式，每行包含一个预测结果：

```json
{"question_id": 504810000, "answer": "yes"}
{"question_id": 504810001, "answer": "sandwich"}
```

支持的字段名：
- `question_id`: 问题ID（必需）
- `answer` 或 `pred`: 模型预测答案（必需）

## 输出格式

评分结果包含以下信息：

- `count`: 评分样本数
- `overall`: 总体软准确率
- `per_question_type`: 按问题类型分类的准确率
- `per_answer_type`: 按答案类型分类的准确率
- `details`: 每个样本的详细评分信息

## 评分算法

评分算法基于VQAv2官方评估工具，实现软准确率计算：

1. 对每个预测答案，与所有标准答案进行比较
2. 计算与其他标准答案的匹配度
3. 使用软准确率公式：min(1, matches/3)

## 注意事项

1. 确保标注文件路径正确：`/data/model/Inference_VLM/VLM_Infra/datasets/VQAv2/annotations/v2_mscoco_val2014_annotations.json`
2. 预测文件必须包含有效的question_id
3. 确保预测文件格式正确（JSONL格式）