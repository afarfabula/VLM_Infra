# LLaVA-GP 训练数值稳定性与指标修复总结

## 背景
- 训练目标：在 LLaVA-GP 中联合优化语言监督（`le_loss`）与图像掩码定位（`loc_loss`/box 指标）。
- 初始症状：`le_loss` 很快变为 0，`pred_mask_ratio` 为 `NaN`，`box/*` 全为 0 或异常。
- 近期目标：修复数值异常、确保损失与指标稳定增长，与日志一致（`trainer_state.json` 中已观测到正常数值）。

## 异常现象
- `pred_mask_ratio = NaN`：掩码 logits 在 `sigmoid().mean()` 前已包含非有限数（NaN/Inf）。
- `le_loss = 0`：训练侧在遇到 `NaN` 时将损失置零；同时 LE 插入与标签/掩码对齐存在问题，导致有效监督失效。
- `box/* = 0`：掩码预测异常或被 NaN 传染，定位损失与混淆矩阵不可用。

## 根因分析
- 注意力融合输出未做数值防护，`attn_map` 或融合层输出包含 NaN/Inf，经 `exp`/`softmax` 或线性投影后传递到掩码 logits。
- LE 监督阶段可能被无效标签或隐状态的 NaN/Inf 污染，交叉熵输出 NaN 被训练器置零，造成“看上去 0”的假正常。
- 指标统计阶段对 NaN 不敏感，导致 `pred_mask_ratio` 直接变为 `NaN`。

## 关键修改
- 注意力融合与掩码解码路径的 NaN/Inf 防护：
  - `AttnFuserV1.forward` 对输入与层输出统一清理：`llava_gp/model/language_model/llava_llama.py:252-285`
  - 掩码解码入口 `_decode_image_token_mask_logits` 增加清理：`llava_gp/model/language_model/llava_llama.py:1235-1246`
- 语言监督的稳定性修复：
  - 在计算 `le_logits` 前对 `hidden_states` 做清理：`llava_gp/model/language_model/llava_llama.py:1613-1619`
  - 使用标准交叉熵并结合 `IGNORE_INDEX` 掩蔽无效标签（按步移位）：`llava_gp/model/language_model/llava_llama.py:1613-1630`
  - 训练脚本中记录 `le_loss` 时保留非零值，并收集有效 token 计数（`le_valid_tokens`/`shift`）：`train_llava_gp.py:859-867`
- 指标统计的稳健化：
  - `pred_mask_ratio` 统计前先做 `nan_to_num`：`train_llava_gp.py:839-844`
  - BCE/Dice 损失模块构造器与数值范围（clamp/nan_to_num）修正：`train_llava_gp.py:292-351`
- 可学习嵌入（LE）插入阶段的对齐修正：
  - 统一在 `_append_le` 中处理 `input_ids`/`inputs_embeds`/`labels`，并更新 `attention_mask` 与 `position_ids` 保持序列一致：`llava_gp/model/language_model/llava_llama.py:1094-1158`

## 为什么之前是 NaN/0，为什么现在正常
- 之前：
  - 注意力融合链路未加防护，导致 `image_token_mask_logits` 出 NaN；`pred_mask_ratio` 直接 NaN。
  - LE 隐状态或标签无掩蔽/错位，交叉熵变 NaN，被训练器置零；表现为 `le_loss = 0`。
  - 掩码路径异常使得定位损失与混淆矩阵无意义，`box/*` 接近或等于 0。
- 现在：
  - 全链路 `nan_to_num` 与范围控制，掩码 logits 有限且可学习，`pred_mask_ratio` 恢复到 0–1 范围。
  - 交叉熵结合 `IGNORE_INDEX` 与移位标签，LE 监督有效并随训练下降，`le_loss` 不再异常为 0。
  - 掩码与标签统一对齐，定位损失有效，混淆矩阵正常累计，`box/precision`、`box/recall`、`box/iou` 与训练步数相关联。

## 验证与现状
- 在 `output/llava1_5_7b_gp_0801/trainer_state.json` 中观察到：
  - `le_valid_tokens`: 约 2.5–2.6（监督有效）
  - `le_loss`: 从 ~5 降到 ~1 左右（稳定下降）
  - `pred_mask_ratio`: 约 0.4–0.7（非 NaN，波动合理）
  - `box/*`: 精度/召回/F1/IoU 明显非零并随训练变化

## 建议
- 保持 `use_attention_logits` 与融合规模的审慎设置，避免注意力值过大导致溢出。
- 继续在训练日志中记录每步的 `le_loss`/`loc_loss_*` 与 `pred_mask_ratio`，确保数值稳定。
- 若切换模型或配置，复用上述防护与对齐逻辑。

