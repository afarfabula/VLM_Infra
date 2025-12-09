import torch
import torch.nn as nn
import os

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            # Resolve local vt_name for cfg_only to avoid remote fetch on delay_load
            def _valid_local_model_dir(p: str) -> bool:
                return bool(p) and os.path.isdir(p) and os.path.isfile(os.path.join(p, 'config.json'))
            vt_name = self.vision_tower_name
            local_override = os.getenv("LOCAL_CLIP_VISION_PATH", "")
            offline = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1"
            if _valid_local_model_dir(local_override):
                vt_name = local_override
            elif _valid_local_model_dir(vt_name):
                vt_name = vt_name
            elif offline:
                default_local = "/data/model/Inference_VLM/models-clip-vit-large-patch14-336"
                if _valid_local_model_dir(default_local):
                    vt_name = default_local
                else:
                    raise RuntimeError(
                        f"Offline mode set; local CLIP path unresolved. Requested '{self.vision_tower_name}'. "
                        f"Set LOCAL_CLIP_VISION_PATH to a local directory containing config.json, "
                        f"or ensure config.mm_vision_tower points to one."
                    )
            print(f"[rank {os.getenv('LOCAL_RANK','0')}] CLIP cfg vt_name -> {vt_name} (offline={offline})")
            self.cfg_only = CLIPVisionConfig.from_pretrained(vt_name, local_files_only=True)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        def _valid_local_model_dir(p: str) -> bool:
            return bool(p) and os.path.isdir(p) and os.path.isfile(os.path.join(p, 'config.json'))

        vt_name = self.vision_tower_name
        local_override = os.getenv("LOCAL_CLIP_VISION_PATH", "")
        offline = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1"

        # Resolve vt_name with strong local preference
        if _valid_local_model_dir(local_override):
            vt_name = local_override
        elif _valid_local_model_dir(vt_name):
            vt_name = vt_name
        elif offline:
            # Last-chance fallback: try default local path for common CLIP-336
            default_local = "/data/model/Inference_VLM/models-clip-vit-large-patch14-336"
            if _valid_local_model_dir(default_local):
                vt_name = default_local
            else:
                raise RuntimeError(
                    f"Offline mode set; local CLIP path unresolved. Requested '{self.vision_tower_name}'. "
                    f"Set LOCAL_CLIP_VISION_PATH to a local directory containing config.json, "
                    f"or ensure config.mm_vision_tower points to one."
                )
        # Log resolved vt_name for debugging
        print(f"[rank {os.getenv('LOCAL_RANK','0')}] CLIP vt_name -> {vt_name} (offline={offline})")

        # Strict local load: build processor and model from local files, then load weights manually
        self.image_processor = CLIPImageProcessor.from_pretrained(vt_name, local_files_only=True)
        cfg = CLIPVisionConfig.from_pretrained(vt_name, local_files_only=True)
        self.vision_tower = CLIPVisionModel(cfg)
        # Find local weight file
        weight_file = None
        for cand in ("pytorch_model.bin", "model.safetensors", "vision_model.safetensors", "vision_model.bin"):
            fp = os.path.join(vt_name, cand)
            if os.path.isfile(fp):
                weight_file = fp
                break
        if weight_file is None:
            raise RuntimeError(
                f"No local CLIP weights found under '{vt_name}'. Expected one of: pytorch_model.bin, model.safetensors."
            )
        # Load state dict and filter to vision_model keys if needed
        if weight_file.endswith('.bin'):
            state = torch.load(weight_file, map_location='cpu')
        else:
            try:
                from safetensors.torch import load_file as safe_load_file
            except ImportError:
                raise ImportError("safetensors is required to load '.safetensors' files locally. Install safetensors or provide .bin weights.")
            state = safe_load_file(weight_file)
        vision_state = {k: v for k, v in state.items() if k.startswith('vision_model')}
        if not vision_state:
            vision_state = state
        missing, unexpected = self.vision_tower.load_state_dict(vision_state, strict=False)
        if missing:
            print(f"[CLIPVisionTower] missing keys: {len(missing)}")
        if unexpected:
            print(f"[CLIPVisionTower] unexpected keys: {len(unexpected)}")
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



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        # Reuse strict local-only loading from base class
        super().load_model(device_map=device_map)

        # Apply S2-specific preprocessing sizes
        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.s2_image_size
        self.image_processor.crop_size['width'] = self.s2_image_size

        # is_loaded is already set in super().load_model

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
