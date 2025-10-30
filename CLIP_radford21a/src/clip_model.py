"""
CLIP (Contrastive Language-Image Pre-training) Model Implementation
Based on: "Learning Transferable Visual Models From Natural Language Supervision" (Radford et al., 2021)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from transformers import AutoModel
import numpy as np

class ImageEncoder(nn.Module):
    """Image encoder using ResNet-50 as backbone"""
    def __init__(self, embed_dim=512, pretrained=True):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Linear(2048, embed_dim)

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        embeddings = self.projection(features)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

class TextEncoder(nn.Module):
    """Text encoder using transformer-based model"""
    def __init__(self, embed_dim=512, model_name='distilbert-base-uncased'):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_dim = self.transformer.config.hidden_size
        self.projection = nn.Linear(hidden_dim, embed_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        embeddings = self.projection(pooled_output)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

class CLIP(nn.Module):
    """CLIP: Contrastive Language-Image Pre-training"""
    def __init__(self, embed_dim=512, temperature=0.07):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim=embed_dim)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

    def forward(self, images, input_ids, attention_mask):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(input_ids, attention_mask)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T

        return logits_per_image, logits_per_text

    def encode_image(self, images):
        return self.image_encoder(images)

    def encode_text(self, input_ids, attention_mask):
        return self.text_encoder(input_ids, attention_mask)

def contrastive_loss(logits_per_image, logits_per_text):
    """Symmetric cross-entropy loss for contrastive learning"""
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)

    return (loss_i + loss_t) / 2

if __name__ == "__main__":
    model = CLIP(embed_dim=512)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
