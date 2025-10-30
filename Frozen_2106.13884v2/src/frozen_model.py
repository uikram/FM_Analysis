"""
Frozen: Multimodal Few-Shot Learning with Frozen Language Models
Based on: Tsimpoukelli et al., 2021
"""
import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoTokenizer, AutoModelForCausalLM

class VisionEncoder(nn.Module):
    """Vision encoder that maps images to language model embedding space"""
    def __init__(self, output_dim=768, num_tokens=2):
        super().__init__()

        # Use the new 'weights' parameter API
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        resnet = models.resnet50(weights=weights)

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 2048
        self.output_dim = output_dim
        self.num_tokens = num_tokens
        self.projection = nn.Linear(self.feature_dim, output_dim * num_tokens)

    def forward(self, x):
        """
        Args:
            x: Images [batch_size, 3, 224, 224]
        Returns:
            Visual prefix: [batch_size, num_tokens, output_dim]
        """
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        projected = self.projection(features)
        batch_size = x.size(0)
        visual_prefix = projected.view(batch_size, self.num_tokens, self.output_dim)
        return visual_prefix

class FrozenModel(nn.Module):
    """Frozen: Vision encoder trained through frozen language model"""
    def __init__(self, lm_model_name='gpt2', num_visual_tokens=2, freeze_lm=True):
        super().__init__()

        self.language_model = AutoModelForCausalLM.from_pretrained(lm_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(lm_model_name)

        if freeze_lm:
            for param in self.language_model.parameters():
                param.requires_grad = False

        embed_dim = self.language_model.config.n_embd
        self.vision_encoder = VisionEncoder(output_dim=embed_dim, num_tokens=num_visual_tokens)
        self.text_embeddings = self.language_model.transformer.wte

    def forward(self, images=None, input_ids=None, attention_mask=None, labels=None):
        """Forward pass with optional image conditioning"""
        batch_size = input_ids.size(0) if input_ids is not None else images.size(0)

        if input_ids is not None:
            text_embeds = self.text_embeddings(input_ids)
        else:
            text_embeds = None

        if images is not None:
            visual_embeds = self.vision_encoder(images)

            if text_embeds is not None:
                inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

                if attention_mask is not None:
                    visual_attention = torch.ones(batch_size, visual_embeds.size(1),
                                                 device=attention_mask.device,
                                                 dtype=attention_mask.dtype)
                    attention_mask = torch.cat([visual_attention, attention_mask], dim=1)

                if labels is not None:
                    visual_labels = torch.full((batch_size, visual_embeds.size(1)),
                                              -100, device=labels.device, dtype=labels.dtype)
                    labels = torch.cat([visual_labels, labels], dim=1)
            else:
                inputs_embeds = visual_embeds
        else:
            inputs_embeds = text_embeds

        outputs = self.language_model(inputs_embeds=inputs_embeds,
                                      attention_mask=attention_mask,
                                      labels=labels)
        return outputs

if __name__ == "__main__":
    model = FrozenModel(lm_model_name='gpt2', num_visual_tokens=2)
    print("--- Frozen Model Test ---")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("-------------------------")
