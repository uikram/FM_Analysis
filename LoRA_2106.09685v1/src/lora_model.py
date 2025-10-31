"""
LoRA (Low-Rank Adaptation) Implementation
Based on: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
"""

import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    """LoRA layer: h = W0*x + (B*A)*x"""
    def __init__(self, in_features, out_features, rank=4, alpha=1, dropout=0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Initialize A with Kaiming uniform and B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x, original_output):
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        lora_output = self.dropout(lora_output)
        return original_output + lora_output

class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation"""
    def __init__(self, linear_layer, rank=4, alpha=1, dropout=0.0):
        super().__init__()
        self.linear = linear_layer
        # Store initialization arguments for model recreation
        self.__init_args__ = (linear_layer,)
        self.__init_kwargs__ = {'rank': rank, 'alpha': alpha, 'dropout': dropout}
        
        # Freeze the original weights
        for param in self.linear.parameters():
            param.requires_grad = False

        self.lora = LoRALayer(
            in_features=linear_layer.in_features,
            out_features=linear_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )

    def forward(self, x):
        original_output = self.linear(x)
        return self.lora(x, original_output)

def apply_lora_to_model(model, rank=4, alpha=1, dropout=0.0, target_modules=['q_proj', 'v_proj']):
    """Apply LoRA to specific modules in a model"""
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            # Check if the module is a Linear layer
            if isinstance(module, nn.Linear):
                # Get the parent module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model
                if parent_name:
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)

                # Replace the linear layer with LoRALinear
                lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
                setattr(parent, child_name, lora_layer)

    return model

# Test if the script is run directly
if __name__ == "__main__":
    # Create a dummy base linear layer
    base_linear = nn.Linear(512, 512)
    # Apply LoRA
    lora_linear = LoRALinear(base_linear, rank=4)

    # Create dummy input
    x = torch.randn(16, 512)
    output = lora_linear(x)

    print("--- LoRA Model Test ---")
    print(f"Base params: {sum(p.numel() for p in base_linear.parameters()):,}")
    print(f"LoRA params: {sum(p.numel() for p in lora_linear.lora.parameters()):,}")
    print(f"Output shape: {output.shape}")
    print("-----------------------")
