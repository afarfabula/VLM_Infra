"""
Patch module to replace the vision tower with a local version that can load models from local paths
"""

import os
import torch
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

def patch_vision_tower(model, local_vision_tower_path):
    """
    Patch the vision tower of the model to use local files
    
    Args:
        model: The LLaVA model
        local_vision_tower_path: Path to the local vision tower model files
    """
    # Get the vision tower from the model
    vision_tower = model.get_vision_tower()
    
    # Check if it's already loaded
    if not vision_tower.is_loaded:
        # Patch the vision tower to use local files
        # We'll monkey patch the load_model method to use local files
        original_load_model = vision_tower.load_model
        
        def patched_load_model(device_map=None):
            """Patched version of load_model that uses local files"""
            if vision_tower.is_loaded:
                print('{} is already loaded, `load_model` called again, skipping.'.format(vision_tower.vision_tower_name))
                return

            # Try to load from local path first
            try:
                if os.path.exists(local_vision_tower_path):
                    print(f"Loading vision tower from local path: {local_vision_tower_path}")
                    vision_tower.image_processor = CLIPImageProcessor.from_pretrained(local_vision_tower_path)
                    vision_tower.vision_tower = CLIPVisionModel.from_pretrained(local_vision_tower_path, device_map=device_map)
                    vision_tower.vision_tower.requires_grad_(False)
                    vision_tower.is_loaded = True
                    return
                else:
                    print(f"Local vision tower path does not exist: {local_vision_tower_path}")
            except Exception as e:
                print(f"Failed to load vision tower from local path: {e}")
                print("Falling back to original method...")
            
            # Fallback to original method
            return original_load_model(device_map)
        
        # Replace the load_model method
        vision_tower.load_model = patched_load_model
        
        # Also patch the config loading for delay_load case
        if hasattr(vision_tower, 'cfg_only'):
            try:
                if os.path.exists(local_vision_tower_path):
                    vision_tower.cfg_only = CLIPVisionConfig.from_pretrained(local_vision_tower_path)
            except Exception as e:
                print(f"Failed to load config from local path: {e}")

def patch_model_for_local_vision_tower(model):
    """
    Patch the model to use local vision tower files
    
    Args:
        model: The LLaVA model
    """
    # Define the local path for the vision tower
    local_vision_tower_path = "/data/model/Inference_VLM/.cache/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
    
    # Apply the patch
    patch_vision_tower(model, local_vision_tower_path)