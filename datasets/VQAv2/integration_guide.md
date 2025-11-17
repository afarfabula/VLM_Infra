# VQAv2多答案评分系统集成指南

## 简介

本文档说明如何将VQAv2多答案评分系统集成到现有的模型评估流程中。

## 集成步骤

### 1. 确保环境依赖

评分系统需要以下Python包：
- jsonlines
- numpy
- argparse

可以通过以下命令安装：
```bash
pip install jsonlines numpy
```

### 2. 准备预测文件

模型推理完成后，需要将预测结果转换为特定格式的JSONL文件：
- 每行一个JSON对象
- 包含question_id和answer字段

示例：
```json
{"question_id": 504810000, "answer": "yes"}
{"question_id": 504810001, "answer": "sandwich"}
```

### 3. 运行评分

可以使用以下任一方式运行评分：

#### 方式一：直接调用Python脚本
```bash
python score_vqav2_with_multiple_gt.py \
    --pred-jsonl /path/to/predictions.jsonl \
    --annotations-json /path/to/v2_mscoco_val2014_annotations.json \
    --output /path/to/scoring_result.json
```

#### 方式二：使用执行脚本
```bash
./run_vqav2_scoring.sh \
    /path/to/predictions.jsonl \
    /path/to/v2_mscoco_val2014_annotations.json \
    /path/to/scoring_result.json
```

### 4. 集成到现有评估流程

#### 修改main.py

在`EvaluatePipeline.run_vqav2_evaluation`方法末尾添加评分调用：

```python
def run_vqav2_evaluation(self):
    # 现有推理代码...
    
    # 合并结果后添加评分
    if self.rank == 0:  # 只在主进程中运行
        # 导入评分模块
        from .score_vqav2_with_multiple_gt import main as score_main
        
        # 准备参数
        pred_file = os.path.join(self.output_dir, "merged_vqav2_inference_results.json")
        annotations_file = "../datasets/VQAv2/annotations/v2_mscoco_val2014_annotations.json"
        output_file = os.path.join(self.output_dir, "vqav2_scoring_results.json")
        
        # 转换预测文件格式
        self._convert_predictions_format(pred_file, pred_file.replace('.json', '.jsonl'))
        
        # 运行评分
        import sys
        from unittest.mock import patch
        with patch('sys.argv', [
            'score_vqav2_with_multiple_gt.py',
            '--pred-jsonl', pred_file.replace('.json', '.jsonl'),
            '--annotations-json', annotations_file,
            '--output', output_file
        ]):
            try:
                score_main()
                print(f"VQAv2评分完成，结果保存在: {output_file}")
            except Exception as e:
                print(f"评分过程中出现错误: {e}")
```

添加辅助方法来转换预测文件格式：

```python
def _convert_predictions_format(self, input_file, output_file):
    """将预测文件从JSON格式转换为JSONL格式"""
    import json
    
    with open(input_file, 'r') as f:
        predictions = json.load(f)
    
    with open(output_file, 'w') as f:
        for pred in predictions:
            # 假设原始预测格式为 {"question_id": xxx, "answer": xxx}
            # 如果格式不同，请相应调整
            f.write(json.dumps({
                "question_id": pred["question_id"],
                "answer": pred["answer"]
            }) + '\n')
    
    print(f"预测文件格式转换完成: {input_file} -> {output_file}")
```

### 5. 自动化运行脚本

创建一个自动化脚本来运行整个评估流程：

```bash
#!/bin/bash
# run_full_evaluation.sh

CONFIG_FILE=$1
MODEL_PATH=$2
OUTPUT_DIR=${3:-"./results"}

echo "开始完整评估流程..."

# 运行推理
echo "1. 运行模型推理..."
python main.py \
    --config $CONFIG_FILE \
    --model $MODEL_PATH \
    --output $OUTPUT_DIR

# 检查推理是否成功
if [ $? -ne 0 ]; then
    echo "模型推理失败!"
    exit 1
fi

# 运行评分
echo "2. 运行VQAv2评分..."
ANNOTATIONS_FILE="../datasets/VQAv2/annotations/v2_mscoco_val2014_annotations.json"
PREDICTIONS_FILE="$OUTPUT_DIR/merged_vqav2_inference_results.jsonl"
SCORING_OUTPUT="$OUTPUT_DIR/vqav2_scoring_results.json"

# 转换预测文件格式（如果尚未转换）
if [ ! -f "$PREDICTIONS_FILE" ]; then
    echo "转换预测文件格式..."
    python -c "
import json
with open('$OUTPUT_DIR/merged_vqav2_inference_results.json', 'r') as f:
    preds = json.load(f)
with open('$PREDICTIONS_FILE', 'w') as f:
    for pred in preds:
        json.dump({'question_id': pred['question_id'], 'answer': pred['answer']}, f)
        f.write('\n')
"
fi

# 运行评分
python score_vqav2_with_multiple_gt.py \
    --pred-jsonl $PREDICTIONS_FILE \
    --annotations-json $ANNOTATIONS_FILE \
    --output $SCORING_OUTPUT

# 检查评分是否成功
if [ $? -eq 0 ]; then
    echo "评估完成! 结果保存在: $SCORING_OUTPUT"
    
    # 显示关键指标
    echo "=== 评估结果摘要 ==="
    python -c "
import json
with open('$SCORING_OUTPUT', 'r') as f:
    results = json.load(f)
print(f'评测样本数: {results[\"count\"]}')
print(f'总体准确率: {results[\"overall\"]:.2f}%')
print('按问题类型:')
for qtype, acc in results['per_question_type'].items():
    print(f'  {qtype}: {acc:.2f}%')
print('按答案类型:')
for atype, acc in results['per_answer_type'].items():
    print(f'  {atype}: {acc:.2f}%')
"
else
    echo "评分过程失败!"
    exit 1
fi
```

使脚本可执行：
```bash
chmod +x run_full_evaluation.sh
```

使用示例：
```bash
./run_full_evaluation.sh configs/vqav2_config.yaml /path/to/model ./results
```

## 注意事项

1. 确保标注文件路径正确：`../datasets/VQAv2/annotations/v2_mscoco_val2014_annotations.json`
2. 确保预测文件中的question_id与标注文件一致
3. 评分系统支持answer和pred两种字段名，推荐使用answer以保持一致性
4. 输出文件包含详细的评分信息，可用于进一步分析

## 故障排除

### 问题1：评分样本数为0
可能原因：
- 预测文件中的question_id与标注文件不匹配
- 预测文件格式不正确

解决方案：
- 检查预测文件中的question_id是否存在于标注文件中
- 确认预测文件为JSONL格式，每行一个JSON对象

### 问题2：找不到标注文件
可能原因：
- 标注文件路径不正确
- 数据集未正确下载

解决方案：
- 确认标注文件路径：`../datasets/VQAv2/annotations/v2_mscoco_val2014_annotations.json`
- 如需重新下载数据集，请参考项目文档

### 问题3：评分脚本报错
可能原因：
- 缺少依赖包
- Python环境问题

解决方案：
- 安装所需依赖：`pip install jsonlines numpy`
- 确认Python版本兼容性（建议Python 3.7+）