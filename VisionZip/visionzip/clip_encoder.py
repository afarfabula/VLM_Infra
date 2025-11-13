#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ------------------------------------------------------------------------
# Modified from LLaVA (https://github.com/haotian-liu/LLaVA)
# Copyright 2024 Senqiao Yang
# ------------------------------------------------------------------------
import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPAttention, CLIPEncoder

from .utils import CLIPAttention_forward, CLIP_EncoderLayer_forward



# ======================= VisionZip 视觉塔压缩逻辑详解 =======================
# 本文件将 LLaVA 的 CLIPVisionTower 的 forward 替换为 VisionZip 版本，核心是：
# - 从 CLIP 编码器读取：倒数第二层的 attentions[-2] 与 hidden_states[-2]；
# - 利用该层在 CLIP_EncoderLayer_forward 暴露的 `metric`（见 utils.py），作为上下文聚类的特征；
# - 两阶段压缩：Dominant 选取 + Contextual 聚合；最终拼接得到压缩后的视觉 token 序列。
# 具体流程：
# 1）Dominant（显著）视觉 token 选择：
#   - 以 CLS 索引（0）为 query，计算其对所有 patch（1..N）的注意力权重之和（跨头累和），
#   - 取 Top-K 索引作为显著 patch 的位置，并保留 CLS，自此得到 dominant_tokens（K+1 个）。
# 2）Contextual（上下文）视觉 token 聚合：
#   - 先从 hidden_states 中剔除 Dominant 所在位置，得到隐藏态 `hidden_states_filtered`；
#   - 对应地从 `metric` 中剔除相同位置并做 L2 归一化，得到 `metric_normalized`；
#   - 以等间隔从 `metric_normalized` 选取 `contextual_num` 个 target 作为聚类中心，
#     其余 token 依据相似度（`similarity = tokens_to_merge @ target_tokens^T`）分配到各中心，
#     并按分配 one-hot 与计数加权，对 hidden_states 做加权平均聚合，得到 `aggregated_hidden`；
#   - 将 target 对应的原始 hidden 与聚合量相加，得到 `contextual_tokens`。
# 3）拼接输出：dominant_tokens + contextual_tokens，作为压缩后的视觉 token，传给 mm_projector。
# 附加输出：all_indices（包含 CLS 与 Dominant patch 的原始索引），用于多图/anyres 情况下的恢复（llava_arch.restore）。
# 注意：list 输入走原始 LLaVA 的 feature_select 流程；仅在 batch 图像上进行 VisionZip 压缩。

class CLIPVisionTower_VisionZip(nn.Module):


    @torch.no_grad()
    def forward(self, images):
        
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True, output_attentions=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, output_attentions=True)
            attn_weights  = image_forward_outs.attentions[-2]
            hidden_states = image_forward_outs.hidden_states[-2]
            # `metric` 由 CLIP_EncoderLayer_forward 在倒数第二层暴露（utils.py），用于上下文聚合
            metric = self.vision_tower.vision_model.encoder.layers[-2].metric
            dominant_num =  self.vision_tower._info["dominant"]
            contextual_num = self.vision_tower._info["contextual"]

            ## Dominant Visual Tokens（基于 CLS 的注意力 Top-K 选取）
            cls_idx = 0
            cls_attention = attn_weights[:, :, cls_idx, cls_idx+1:]  
            cls_attention_sum = cls_attention.sum(dim=1)  
            topk_indices = cls_attention_sum.topk(dominant_num, dim=1).indices + 1
            # all_indices 保留 CLS(0) 与 Top-K patch 的原始位置，用于后续恢复
            all_indices = torch.cat([torch.zeros((hidden_states.shape[0], 1), dtype=topk_indices.dtype, device=topk_indices.device), topk_indices], dim=1)
            
            mask = torch.ones_like(hidden_states[:, :, 0], dtype=torch.bool, device=metric.device).scatter_(1, all_indices, False)
            dominant_tokens = hidden_states.masked_select(~mask.unsqueeze(-1)).view(hidden_states.shape[0], dominant_num + 1, hidden_states.shape[2])
            
            ### Filter（把 Dominant 去除后保留其余 token 与度量）
            metric_filtered = metric[mask].view(hidden_states.shape[0], hidden_states.shape[1] - (dominant_num + 1), metric.shape[2])

            hidden_states_filtered = hidden_states.masked_select(mask.unsqueeze(-1)).view(hidden_states.shape[0], hidden_states.shape[1] - (dominant_num +1), hidden_states.shape[2])  
            
            metric_normalized = metric_filtered / metric_filtered.norm(dim=-1, keepdim=True) 

            ## Contextual Visual Tokens（按 metric 归一化向量做分簇与聚合）
            step = max(1, metric_normalized.shape[1] // contextual_num)
            target_indices = torch.arange(0, metric_normalized.shape[1], step, device=metric_normalized.device)[:contextual_num]
            target_tokens = metric_normalized[:, target_indices, :]

            tokens_to_merge = metric_normalized[:, ~torch.isin(torch.arange(metric_normalized.shape[1], device=metric_normalized.device), target_indices), :]
            similarity = torch.bmm(tokens_to_merge, target_tokens.transpose(1, 2))
            assign_one_hot = torch.zeros(tokens_to_merge.shape[0], tokens_to_merge.shape[1], contextual_num, dtype=hidden_states_filtered.dtype, device=metric_normalized.device)
            assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)
            counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
            hidden_to_merge = hidden_states_filtered[:, ~torch.isin(torch.arange(hidden_states_filtered.shape[1], device=hidden_states_filtered.device), target_indices), :]
            aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2), hidden_to_merge) / counts
            target_hidden = hidden_states_filtered[:, target_indices, :]  
            
            contextual_tokens = target_hidden + aggregated_hidden

            # Merge with target hidden states and concatenate
            hidden_states_save = torch.cat([dominant_tokens, contextual_tokens], dim=1).to(images.dtype)

        # 返回压缩后的 tokens 与选中索引，用于 mm_projector 以及 anyres 的 unpad 还原
        return hidden_states_save, all_indices

        # return hidden_states_save, hidden_states, all_indices





