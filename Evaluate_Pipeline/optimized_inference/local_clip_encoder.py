import torch
import torch.nn as nn
import os
import json
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class LocalCLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        # Support for local model path
        self.vision_tower_name = vision_tower
        self.local_model_path = self._get_local_model_path(vision_tower)
        
        # 修复：处理缺失的 mm_vision_select_layer 属性
        self.select_layer = getattr(args, 'mm_vision_select_layer', -2)  # 默认值 -2
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            # Try to load config from local path first
            try:
                self.cfg_only = self._load_config_from_local(delay_load=True)
            except Exception as e:
                print(f"Failed to load config in delay_load mode: {e}")
                # 在延迟加载模式下，我们只需要一个基本的配置对象
                self.cfg_only = CLIPVisionConfig()

    def _get_local_model_path(self, vision_tower_name):
        """Get local model path for vision tower"""
        if vision_tower_name == "openai/clip-vit-large-patch14-336":
            return "/data/model/Inference_VLM/.cache/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
        return None

    def _load_config_from_local(self, delay_load=False):
        """Try to load config from local path first, fallback to HuggingFace"""
        if delay_load:
            # 在延迟加载模式下，我们尝试从本地路径加载配置
            if self.local_model_path and os.path.exists(self.local_model_path):
                config_path = os.path.join(self.local_model_path, 'config.json')
                if os.path.exists(config_path):
                    try:
                        # Load config from local file
                        return CLIPVisionConfig.from_pretrained(config_path)
                    except Exception as e:
                        print(f"Failed to load config from local path {config_path}: {e}")
                else:
                    # If config.json doesn't exist, try to create it from preprocessor_config.json
                    preprocessor_path = os.path.join(self.local_model_path, 'preprocessor_config.json')
                    if os.path.exists(preprocessor_path):
                        try:
                            with open(preprocessor_path, 'r') as f:
                                preprocessor_config = json.load(f)
                            
                            # Create a basic CLIP config from preprocessor config
                            config_dict = {
                                "model_type": "clip_vision_model",
                                "hidden_size": preprocessor_config.get("hidden_size", 1024),
                                "intermediate_size": preprocessor_config.get("intermediate_size", 4096),
                                "num_attention_heads": preprocessor_config.get("num_attention_heads", 16),
                                "num_hidden_layers": preprocessor_config.get("num_hidden_layers", 24),
                                "image_size": preprocessor_config.get("image_size", 336),
                                "patch_size": preprocessor_config.get("patch_size", 14),
                            }
                            
                            return CLIPVisionConfig(**config_dict)
                        except Exception as e:
                            print(f"Failed to create config from preprocessor config: {e}")
            
            # 如果本地路径不可用，返回一个带有默认值的配置对象
            # 稍后会在load_model中真正加载
            default_config = CLIPVisionConfig(
                hidden_size=1024,
                intermediate_size=4096,
                num_attention_heads=16,
                num_hidden_layers=24,
                image_size=336,
                patch_size=14,
                model_type="clip_vision_model"
            )
            return default_config
            
        if self.local_model_path and os.path.exists(self.local_model_path):
            config_path = os.path.join(self.local_model_path, 'config.json')
            if os.path.exists(config_path):
                try:
                    # Load config from local file
                    return CLIPVisionConfig.from_pretrained(config_path)
                except Exception as e:
                    print(f"Failed to load config from local path {config_path}: {e}")
            else:
                # If config.json doesn't exist, try to create it from preprocessor_config.json
                preprocessor_path = os.path.join(self.local_model_path, 'preprocessor_config.json')
                if os.path.exists(preprocessor_path):
                    try:
                        with open(preprocessor_path, 'r') as f:
                            preprocessor_config = json.load(f)
                        
                        # Create a basic CLIP config from preprocessor config
                        config_dict = {
                            "model_type": "clip_vision_model",
                            "hidden_size": preprocessor_config.get("hidden_size", 1024),
                            "intermediate_size": preprocessor_config.get("intermediate_size", 4096),
                            "num_attention_heads": preprocessor_config.get("num_attention_heads", 16),
                            "num_hidden_layers": preprocessor_config.get("num_hidden_layers", 24),
                            "image_size": preprocessor_config.get("image_size", 336),
                            "patch_size": preprocessor_config.get("patch_size", 14),
                        }
                        
                        return CLIPVisionConfig(**config_dict)
                    except Exception as e:
                        print(f"Failed to create config from preprocessor config: {e}")
        
        # Fallback to original method
        return CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def _load_processor_from_local(self):
        """Load processor from local path if available"""
        if self.local_model_path and os.path.exists(self.local_model_path):
            try:
                processor = CLIPImageProcessor.from_pretrained(self.local_model_path)
                print(f"Loaded processor from local path: {self.local_model_path}")
                return processor
            except Exception as e:
                print(f"Failed to load processor from local path: {e}")
        
        # Fallback to original method
        return CLIPImageProcessor.from_pretrained(self.vision_tower_name)

    def _load_model_from_local(self, device_map=None):
        """Load model from local path if available"""
        if self.local_model_path and os.path.exists(self.local_model_path):
            try:
                model = CLIPVisionModel.from_pretrained(self.local_model_path, device_map=device_map)
                print(f"Loaded model from local path: {self.local_model_path}")
                return model
            except Exception as e:
                print(f"Failed to load model from local path: {e}")
        
        # Fallback to original method
        return CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        # Load from local path first
        self.image_processor = self._load_processor_from_local()
        self.vision_tower = self._load_model_from_local(device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

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