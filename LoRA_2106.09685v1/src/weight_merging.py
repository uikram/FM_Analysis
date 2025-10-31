"""
Weight Merging Utilities for LoRA
Implements W_merged = W_frozen + BA functionality for inference optimization
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from lora_model import LoRALinear

def merge_lora_weights(model) -> Dict[str, Any]:
    """
    Merge LoRA weights with frozen weights for faster inference.
    Returns a state dict with merged weights.
    """
    merged_state = {}
    original_state = model.state_dict()

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # Get the base weights
            base_weight = module.linear.weight
            
            # Calculate LoRA contribution (BA)
            lora_A = module.lora.lora_A  # [r x in_features]
            lora_B = module.lora.lora_B  # [out_features x r]
            scaling = module.lora.scaling
            
            # Merge: W = W_frozen + scaling * BA
            merged_weight = base_weight + (lora_B @ lora_A) * scaling
            
            # Replace weight in state dict with merged version
            parent_name = name if name else ''
            weight_name = f"{parent_name}.linear.weight"
            merged_state[weight_name] = merged_weight

    # Copy all other weights as-is
    for key, value in original_state.items():
        if key not in merged_state:
            merged_state[key] = value

    return merged_state

def create_merged_model(model):
    """
    Create a new model instance with merged weights for faster inference.
    Original model remains unchanged.
    """
    # Create a new model instance
    new_model = type(model)(*model.__init_args__, **model.__init_kwargs__)
    
    # Merge weights
    merged_state = merge_lora_weights(model)
    
    # Load merged weights into new model
    new_model.load_state_dict(merged_state)
    
    return new_model