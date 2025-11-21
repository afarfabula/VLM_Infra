"""
本地视觉塔模型加载模块
用于在离线环境中加载视觉塔模型
"""

import os
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from pathlib import Path


class LocalCLIPVisionTower(nn.Module):
    def __init__(self, vision_tower_path, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_path = vision_tower_path
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            # 尝试从本地路径加载配置
            self.cfg_only = self._load_local_config()

    def _load_local_config(self):
        """从本地路径加载配置"""
        try:
            config_path = Path(self.vision_tower_path) / "config.json"
            if config_path.exists():
                return CLIPVisionConfig.from_json_file(str(config_path))
            else:
                # 如果本地配置不存在，使用默认配置
                return CLIPVisionConfig(
                    hidden_size=1024,
                    intermediate_size=4096,
                    num_hidden_layers=24,
                    num_attention_heads=16,
                    image_size=336,
                    patch_size=14
                )
        except Exception as e:
            print(f"Warning: Failed to load local config: {e}")
            # 返回默认配置
            return CLIPVisionConfig(
                hidden_size=1024,
                intermediate_size=4096,
                num_hidden_layers=24,
                num_attention_heads=16,
                image_size=336,
                patch_size=14
            )

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_path))
            return

        try:
            # 尝试从本地路径加载图像处理器和视觉塔模型
            processor_path = Path(self.vision_tower_path) / "preprocessor_config.json"
            if processor_path.exists():
                self.image_processor = CLIPImageProcessor.from_json_file(str(processor_path))
            else:
                # 使用默认的图像处理器
                self.image_processor = CLIPImageProcessor(
                    do_resize=True,
                    size={"shortest_edge": 336},
                    resample=3,
                    do_center_crop=True,
                    crop_size=336,
                    do_normalize=True,
                    image_mean=[0.48145466, 0.4578275, 0.40821073],
                    image_std=[0.26862954, 0.26130258, 0.27577711],
                    do_convert_rgb=True,
                )

            # 加载视觉塔模型
            self.vision_tower = CLIPVisionModel.from_pretrained(
                self.vision_tower_path, 
                device_map=device_map,
                local_files_only=True  # 强制只使用本地文件
            )
            self.vision_tower.requires_grad_(False)

            self.is_loaded = True
            print(f"Successfully loaded local vision tower from {self.vision_tower_path}")

        except Exception as e:
            print(f"Failed to load local vision tower: {e}")
            # 如果本地加载失败，尝试在线加载（在有网络的情况下）
            try:
                self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_path)
                self.vision_tower = CLIPVisionModel.from_pretrained(
                    self.vision_tower_path, 
                    device_map=device_map
                )
                self.vision_tower.requires_grad_(False)
                self.is_loaded = True
                print(f"Successfully loaded vision tower from HuggingFace: {self.vision_tower_path}")
            except Exception as e2:
                print(f"Failed to load vision tower from HuggingFace: {e2}")
                raise e2

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2