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

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
import torch.nn.functional as F


# ======================= VisionZip 接入 LLaVA 的数据流注释 =======================
# 本文件通过重写 LLaVA 的多模态准备流程，接入 VisionZip 的视觉压缩：
# - encode_images_visionzip：单图编码，兼容视觉塔 forward 返回 tuple 或单值；
# - encode_images_visionzip_multi：多图/拼接编码，返回压缩后的特征与保留索引（keep_idx）；
# - restore_image_features_sorted：当 mm_patch_merge_type 包含 'unpad' 时，将任意分辨率图像的特征按照保留索引还原成网格并选择；
# - prepare_inputs_labels_for_multimodal_visionzip：把图像特征插入到文本 token 序列中，生成最终的嵌入与标签。
# 设计要点：
# 1）视觉塔 forward 返回 `(hidden_states_save, all_indices)`；在多图情况下，`keep_idxs` 被 split 并参与 anyres 恢复。
# 2）如果配置为 spatial 合并且启用 'unpad'，使用 `restore_image_features_sorted` 按网格大小对齐并拼接；
# 3）其它情况保持与原 LLaVA 一致，仅替换编码入口为 VisionZip。

def encode_images_visionzip(self, images):
    # 单图编码入口：
    # - 读取视觉塔 forward 输出，如果是二元组则取第一个作为压缩后的视觉 tokens；
    # - 通过 mm_projector 映射到 LLM 语言空间（维度与对齐）。
    out = self.get_model().get_vision_tower().forward(images)
    image_features = out[0] if isinstance(out, tuple) else out
    image_features = self.get_model().mm_projector(image_features)
    return image_features

def encode_images_visionzip_multi(self, images):
    # 多图编码入口：
    # - 视觉塔 forward 返回 `(features, keep_idx)`，若为单值则 keep_idx 置为 None；
    # - 经 mm_projector 映射后返回，供后续 split 与 anyres 恢复使用。
    out = self.get_model().get_vision_tower().forward(images)
    if isinstance(out, tuple):
        image_features, keep_idx = out
    else:
        image_features, keep_idx = out, None
    image_features = self.get_model().mm_projector(image_features)
    return image_features, keep_idx 
    

def restore_image_features_sorted(self, image_feature, cur_keep_idx, width, height):
    # 任意分辨率（anyres）'unpad' 情况下的还原：
    # - 输入是每张图的压缩后 tokens（含 CLS 与 Dominant/Contextual），以及对应的保留索引 cur_keep_idx；
    # - 目标是在固定 24x24 网格上复原被保留的 patch，再拼接 CLS 与额外的 raw 特征，供文本拼接使用。
    
    num_img, total_patches, feature_dim = image_feature.shape
    num_keep = cur_keep_idx.shape[1]  
    num_extra = total_patches - num_keep  

    # 按行排序保留索引，并去掉 CLS 的 0，占位偏移 -1 得到真实 patch 索引
    cur_keep_idx_sorted, _ = cur_keep_idx.sort(dim=1)  # [num_img, num_keep]
    cur_keep_idx_sorted_restore = cur_keep_idx_sorted[:, 1:]-1

    # 建网格并用 mask 标出保留位置，填充对应特征
    restored_features = torch.zeros((num_img, 576, feature_dim), device=image_feature.device, dtype=image_feature.dtype)  # [num_img, total_patches, feature_dim]

    mask = torch.zeros(num_img, 576, dtype=torch.bool, device=image_feature.device)
    mask.scatter_(1, cur_keep_idx_sorted_restore, True)  

    kept_features = image_feature[:, 1:num_keep, :]  
    restored_features[mask] = kept_features.reshape(-1, feature_dim)  
    
    # 将 [H,W,24,24] 的拼接顺序转置至 [H,24,W,24] 再 flatten 成 2D 网格
    assert width * height == restored_features.shape[0], "width * height must equal num_img"
    restored_features = restored_features.view(height, width, 24, 24, feature_dim)  # [height, width, 24, 24, feature_dim]
    restored_features = restored_features.permute(0, 2, 1, 3, 4).contiguous()  # [height, 24, width, 24, feature_dim]
    restored_features = restored_features.view(height, 24, width * 24, feature_dim)  # [height, 24, width*24, feature_dim]
    restored_features = restored_features.view(height * 24, width * 24, feature_dim)  # [height*24, width*24, feature_dim]
    image_newline_expanded = self.model.image_newline.view(1, 1, feature_dim).expand(height * 24, 1, feature_dim).to(restored_features.device)  # [height*24, 1, feature_dim]
    grid_with_newline = restored_features

    mask = mask.view(height, width, 24, 24)  # [height, width, 24, 24]
    mask = mask.permute(0, 2, 1, 3).contiguous()  # [height, 24, width, 24]
    mask = mask.view(height * 24, width * 24)  # [height*24, width*24]

    mask_all = mask

    # 从网格中选出保留的 patch 特征，并拼接 CLS 与 raw 的额外特征
    image_feature_select = grid_with_newline[mask_all]
    raw_img_feature_merge = image_feature[:,-num_extra:,].reshape(-1, feature_dim)
    cls_img_feature_merge = image_feature[:,0,]

    image_feature_select = torch.cat([image_feature_select, cls_img_feature_merge, raw_img_feature_merge])
    return image_feature_select

    

def prepare_inputs_labels_for_multimodal_visionzip(
    self, input_ids, position_ids, attention_mask, past_key_values, labels,
    images, image_sizes=None
):
    # 入口：将 VisionZip 视觉特征插入到文本序列中，生成新的嵌入与标签
    vision_tower = self.get_vision_tower()
    if vision_tower is None or images is None or input_ids.shape[1] == 1:
        return input_ids, position_ids, attention_mask, past_key_values, None, labels

    if type(images) is list or images.ndim == 5:
        # 多图（list 或 5D batch）
        if type(images) is list:
            images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
        concat_images = torch.cat([image for image in images], dim=0)
        image_features, keep_idxs = self.encode_images_visionzip_multi(concat_images)
        # 将编码特征按原图 batch 维拆分
        split_sizes = [image.shape[0] for image in images]
        image_features = torch.split(image_features, split_sizes, dim=0)

        keep_idxs= torch.split(keep_idxs, split_sizes, dim=0)
        mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
        image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
        if mm_patch_merge_type == 'flat':
            # 扁平化所有空间维度，保持与原 LLaVA 一致
            image_features = [x.flatten(0, 1) for x in image_features]
        elif mm_patch_merge_type.startswith('spatial'):
            # 空间保留策略：可选含 'unpad' 的 anyres 还原
            new_image_features = []
            for image_idx, image_feature in enumerate(image_features):
                if image_feature.shape[0] > 1:

                    base_image_feature = image_feature[0]
                    image_feature = image_feature[1:]
                    cur_keep_idx = keep_idxs[image_idx][1:]
                    num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)

                    if 'unpad' in mm_patch_merge_type and cur_keep_idx is not None:
                        # 将保留的 patch 还原到 24x24 网格并选取，再拼接回 base
                        image_feature = self.restore_image_features_sorted(image_feature, cur_keep_idx,num_patch_width, num_patch_height)

                    else:
                        # 非 unpad：直接按 [B,C,W,H] → flatten 的传统路径
                        image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                        image_feature = image_feature.flatten(0, 3)
                    image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                else:
                    image_feature = image_feature[0]
                    if 'unpad' in mm_patch_merge_type:
                        image_feature = torch.cat((
                            image_feature,
                            self.model.image_newline[None].to(image_feature.device)
                        ), dim=0)
                new_image_features.append(image_feature)
            image_features = new_image_features
        else:
            raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
    else:
        # 单图：直接调用 VisionZip 的单图编码
        image_features = self.encode_images_visionzip(images)


    # 注：image start / end 在此不实现以支持预训练配置
    if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
        raise NotImplementedError

    # 下面保持与原 LLaVA 的文本-图像拼接逻辑一致：
    _labels = labels
    _position_ids = position_ids
    _attention_mask = attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    if position_ids is None:
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    # 去掉 pad（用 attention_mask）
    _input_ids = input_ids
    input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
    labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

    new_input_embeds = []
    new_labels = []
    cur_image_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        if num_images == 0:
            # 无图：保留原始文本嵌入
            cur_image_features = image_features[cur_image_idx]
            cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
            cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
            new_input_embeds.append(cur_input_embeds)
            new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            continue

        # 将文本按图像占位符分段，交替插入视觉特征
        image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
        cur_input_ids_noim = []
        cur_labels = labels[batch_idx]
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
            cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
        split_sizes = [x.shape[0] for x in cur_labels_noim]
        cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
        cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
        cur_new_input_embeds = []
        cur_new_labels = []

        for i in range(num_images + 1):
            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            cur_new_labels.append(cur_labels_noim[i])
            if i < num_images:
                cur_image_features = image_features[cur_image_idx]
                cur_image_idx += 1
                cur_new_input_embeds.append(cur_image_features)
                cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

        cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

        cur_new_input_embeds = torch.cat(cur_new_input_embeds)
        cur_new_labels = torch.cat(cur_new_labels)

        new_input_embeds.append(cur_new_input_embeds)
        new_labels.append(cur_new_labels)

    # 截断到 tokenizer 最大长度，以防因图像嵌入变长
    tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
    if tokenizer_model_max_length is not None:
        new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
        new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

    # padding、mask 与 position_ids 的整合（与原 LLaVA 一致）
    max_len = max(x.shape[0] for x in new_input_embeds)
    batch_size = len(new_input_embeds)

    new_input_embeds_padded = []
    new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

    for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
        cur_len = cur_new_embed.shape[0]
        if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
            new_input_embeds_padded.append(torch.cat((
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                cur_new_embed
            ), dim=0))
            if cur_len > 0:
                new_labels_padded[i, -cur_len:] = cur_new_labels
                attention_mask[i, -cur_len:] = True
                position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
        else:
            new_input_embeds_padded.append(torch.cat((
                cur_new_embed,
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
            ), dim=0))
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

    new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

    if _labels is None:
        new_labels = None
    else:
        new_labels = new_labels_padded

    if _attention_mask is None:
        attention_mask = None
    else:
        attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    if _position_ids is None:
        position_ids = None

    return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


 