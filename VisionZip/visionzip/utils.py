import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from typing import Any, Optional, Tuple, Union, List
from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPAttention, CLIPVisionTransformer, CLIPEncoder
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput


# ======================= VisionZip Token Reduction 补丁说明 =======================
# 目标：在不改动上游权重的情况下，对 CLIP 视觉编码器的中间层进行“选择与聚合”，
# 将原始 patch tokens 压缩为两类：
# 1）Dominant（显著）视觉 token：由 CLS 对各 patch 的注意力强度挑选出的 Top-K。
# 2）Contextual（上下文）视觉 token：在剩余 token 上做归一化度量并聚类汇聚得到。
# 关键做法：
# - 通过替换 CLIPAttention.forward，附加返回第三项 `metric`（来自 raw_key_states 的跨头平均），
#   作为每个 token 的稳定特征度量，用于下游的聚合。
# - 通过替换 CLIPEncoderLayer.forward，在指定层（r>0）把 `metric` 暴露到模块对象上，
#   后续在视觉塔的 forward 中读取该层的 `metric`。
# - 通过 `apply_info` 向 CLIP 模型注入信息：dominant/contextual 数量与 r 调度，
#   并让所有 CLIPEncoderLayer 共享这份 `_info`。
# 整体数据流：
# 输入图像 → CLIP 编码器（保留 attentions/hidden_states） → 读取 penultimate 层的 attentions 与 metric →
# Top-K 选显著 token（含 CLS） → 在剩余 token 上做基于 `metric` 的分簇与汇聚 →
# 拼接 Dominant + Contextual → 经 mm_projector 映射到 LLM 空间 → 参与多模态对齐与生成。
def CLIPAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    causal_attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""

    bsz, tgt_len, embed_dim = hidden_states.size()

    # get query proj
    query_states = self.q_proj(hidden_states) * self.scale
    key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    raw_key_states = key_states.clone()
    value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

    proj_shape = (bsz * self.num_heads, -1, self.head_dim)
    query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    key_states = key_states.view(*proj_shape)
    value_states = value_states.view(*proj_shape)

    src_len = key_states.size(1)
    attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

    if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
            f" {attn_weights.size()}"
        )

    # apply the causal_attention_mask first
    if causal_attention_mask is not None:
        if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                f" {causal_attention_mask.size()}"
            )
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, tgt_len, src_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)



    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    if output_attentions:
        # this operation is a bit akward, but it's required to
        # make sure that attn_weights keeps its gradient.
        # In order to do so, attn_weights have to reshaped
        # twice and have to be reused in the following
        attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
    else:
        attn_weights_reshaped = None

    attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

    attn_output = torch.bmm(attn_probs, value_states)

    if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

    attn_output = self.out_proj(attn_output)

    # 这里的第三个返回值 `raw_key_states.mean(1)` 即 VisionZip 的度量 `metric`：
    # - 先计算未变形的 key（raw_key_states），按注意力头维度取平均，得到每个 token 的稳定向量表示。
    # - 在指定层（见 CLIP_EncoderLayer_forward）把该 `metric` 暴露给后续步骤使用。
    return attn_output, attn_weights_reshaped, raw_key_states.mean(1)

def CLIP_EncoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    causal_attention_mask: torch.Tensor,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.FloatTensor]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`): attention mask of size
            `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            `(config.encoder_attention_heads,)`.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
    """
    residual = hidden_states

    hidden_states = self.layer_norm1(hidden_states)


    hidden_states, attn_weights, metric = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
    )
    
    hidden_states = residual + hidden_states
    
    r = self._info["r"].pop(0)
    if r > 0:
        self.metric = metric    
    residual = hidden_states
    hidden_states = self.layer_norm2(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (attn_weights,)

    return outputs


def parse_r(num_layers: int, r: Union[List[int], Tuple[int, float], int]) -> List[int]:
    """
    Copy from the TOME. 
    https://github.com/facebookresearch/ToMe

    Process a constant r or r schedule into a list for use internally.

    r can take the following forms:
     - int: A constant number of tokens per layer.
     - Tuple[int, float]: A pair of r, inflection.
       Inflection describes there the the reduction / layer should trend
       upward (+1), downward (-1), or stay constant (0). A value of (r, 0)
       is as providing a constant r. (r, -1) is what we describe in the paper
       as "decreasing schedule". Any value between -1 and +1 is accepted.
     - List[int]: A specific number of tokens per layer. For extreme granularity.
    """
    inflect = 0
    if isinstance(r, list):
        if len(r) < num_layers:
            r = r + [0] * (num_layers - len(r))
        return list(r)
    elif isinstance(r, tuple):
        r, inflect = r

    min_val = int(r * (1.0 - inflect))
    max_val = 2 * r - min_val
    step = (max_val - min_val) / (num_layers - 1)

    return [int(min_val + step * i) for i in range(num_layers)]

def make_tome_class(transformer_class):
    # VisionZip 包装 CLIPVisionTransformer：
    # - 在每次 forward 前把 `self._info['r']` 解析为逐层调度（parse_r），
    # - 初始化 `_info['size']` / `_info['source']`（保持兼容），
    # - 使所有下游的 CLIPEncoderLayer 都能访问同一份 `_info`。
    class VisionZipTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """
        # 注：这里不改具体计算，只在 forward 前准备好 `_info`，为层内逻辑提供上下文。
            
        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._info["r"] = parse_r(len(self.vision_model.encoder.layers), self.r)
            # self._info["r"] = self.r

            self._info["size"] = None
            self._info["source"] = None

            return super().forward(*args, **kwdargs)

    return VisionZipTransformer

def apply_info(model, dominant_num, contextual_num):
    # VisionZip 入口：将补丁注入到已加载的 CLIP 视觉模型
    # - 用 make_tome_class 包装模型类，实现 `_info` 的共享与 r 调度。
    # - 设置 r 调度：`[0]*22 + [1] + [0]`，即只在倒数第二层启用 metric 导出。
    # - 注入 `_info` 的两个关键参数：
    #   `dominant`（显著 token 数，含 CLS）与 `contextual`（上下文 token 数）。
    # - 将 `_info` 绑定到每个 CLIPEncoderLayer，使其在 forward 中按 r 调度暴露 metric。
    # 注意：dominant 在视觉塔 forward 会通过 CLS 注意力 Top-K 选出；contextual 则基于 metric 聚合。

    VisionZipTransformer = make_tome_class(model.__class__)

    model.__class__ = VisionZipTransformer
    model.r = [0 for i in range(22)]+ [1]+[0]

    model._info = {
        "r": [model.r],
        "dominant":dominant_num,
        "contextual":contextual_num,
    }
    for module in model.modules():
        if isinstance(module, CLIPEncoderLayer):
            module._info = model._info

