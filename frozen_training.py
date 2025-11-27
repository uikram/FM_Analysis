"""
Frozen: Multimodal Few-Shot Learning with Frozen Language Models
Complete Training Pipeline Implementation

Based on: Tsimpoukelli et al., 2021
https://arxiv.org/abs/2106.13884

This implementation provides both full-scale and RTX A5000-compatible versions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPTNeoForCausalLM
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
from typing import Optional, Tuple, List
import json
from pathlib import Path
from tqdm import tqdm
import wandb
from dataclasses import dataclass


@dataclass
class FrozenConfig:
    """Configuration for Frozen model"""
    # Model architecture
    vision_encoder_name: str = "resnet50"  # Using standard ResNet50 as proxy for NF-ResNet50
    language_model_name: str = "gpt2-large"  # 774M params (fits A5000)
    # For full scale: use "EleutherAI/gpt-neo-2.7B" or larger
    
    visual_prefix_length: int = 2  # Number of visual tokens
    vision_hidden_dim: int = 2048  # ResNet50 output dim
    lm_hidden_dim: int = 1280  # GPT2-large hidden size (1024 for gpt2, 1280 for gpt2-large)
    
    # Training hyperparameters
    batch_size: int = 8  # Reduced for A5000
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    num_epochs: int = 3  # Paper uses early stopping after ~1 epoch
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 16  # Effective batch size = 128
    
    # Data
    image_size: int = 224
    max_caption_length: int = 128
    
    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    fp16: bool = True  # Mixed precision training
    
    # Paths
    output_dir: str = "./frozen_outputs"
    checkpoint_dir: str = "./frozen_checkpoints"
    

class VisionEncoder(nn.Module):
    """Vision encoder using ResNet-50 (proxy for NF-ResNet-50)"""
    
    def __init__(self, config: FrozenConfig):
        super().__init__()
        self.config = config
        
        # Load pretrained ResNet50
        # Note: Paper uses NF-ResNet-50, but standard ResNet50 is a good proxy
        self.backbone = timm.create_model(
            config.vision_encoder_name,
            pretrained=True,
            num_classes=0,  # Remove classification head
            global_pool=''  # We'll do our own pooling
        )
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Project to visual prefix: vision_dim -> (prefix_length * lm_hidden_dim)
        self.projection = nn.Linear(
            config.vision_hidden_dim,
            config.visual_prefix_length * config.lm_hidden_dim
        )
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [batch_size, 3, 224, 224]
        Returns:
            visual_prefix: [batch_size, prefix_length, lm_hidden_dim]
        """
        # Extract features
        features = self.backbone(images)  # [B, 2048, 7, 7]
        
        # Global pooling
        features = self.pool(features)  # [B, 2048, 1, 1]
        features = features.flatten(1)  # [B, 2048]
        
        # Project and reshape to visual prefix
        prefix = self.projection(features)  # [B, prefix_length * lm_hidden_dim]
        prefix = prefix.view(
            -1, 
            self.config.visual_prefix_length, 
            self.config.lm_hidden_dim
        )  # [B, prefix_length, lm_hidden_dim]
        
        return prefix


class FrozenModel(nn.Module):
    """
    Frozen: Vision encoder + Frozen language model
    Only the vision encoder is trained.
    """
    
    def __init__(self, config: FrozenConfig):
        super().__init__()
        self.config = config
        
        # Initialize vision encoder (trainable)
        self.vision_encoder = VisionEncoder(config)
        
        # Initialize frozen language model
        print(f"Loading language model: {config.language_model_name}")
        if "gpt-neo" in config.language_model_name.lower():
            self.language_model = GPTNeoForCausalLM.from_pretrained(
                config.language_model_name
            )
        else:
            self.language_model = GPT2LMHeadModel.from_pretrained(
                config.language_model_name
            )
        
        # Freeze language model
        for param in self.language_model.parameters():
            param.requires_grad = False
        
        print(f"Language model frozen. Total params: {sum(p.numel() for p in self.language_model.parameters()):,}")
        print(f"Vision encoder trainable params: {sum(p.numel() for p in self.vision_encoder.parameters() if p.requires_grad):,}")
        
    def forward(
        self, 
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through Frozen model
        
        Args:
            images: [batch_size, 3, 224, 224]
            input_ids: [batch_size, seq_len] - tokenized captions
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len] - for computing loss
            
        Returns:
            logits: [batch_size, prefix_len + seq_len, vocab_size]
            loss: scalar (if labels provided)
        """
        batch_size = images.size(0)
        
        # Get visual prefix embeddings
        visual_prefix = self.vision_encoder(images)  # [B, prefix_len, hidden_dim]
        
        # Get text embeddings
        text_embeds = self.language_model.transformer.wte(input_ids)  # [B, seq_len, hidden_dim]
        
        # Concatenate visual prefix and text embeddings
        combined_embeds = torch.cat([visual_prefix, text_embeds], dim=1)  # [B, prefix_len + seq_len, hidden_dim]
        
        # Create attention mask for combined sequence
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        
        # Extend attention mask for visual prefix (always attend)
        prefix_attention = torch.ones(
            (batch_size, self.config.visual_prefix_length),
            dtype=torch.long,
            device=images.device
        )
        combined_attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)
        
        # Forward through frozen language model
        outputs = self.language_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            labels=None,  # We'll compute loss manually
            return_dict=True
        )
        
        logits = outputs.logits  # [B, prefix_len + seq_len, vocab_size]
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for causal LM loss
            # Only compute loss on text tokens, not visual prefix
            shift_logits = logits[:, self.config.visual_prefix_length:-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Flatten for loss computation
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return logits, loss
    
    def generate(
        self,
        images: torch.Tensor,
        tokenizer,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> List[str]:
        """Generate captions for images"""
        self.eval()
        batch_size = images.size(0)
        
        with torch.no_grad():
            # Get visual prefix
            visual_prefix = self.vision_encoder(images)
            
            # Start with empty text (BOS token)
            input_ids = torch.full(
                (batch_size, 1),
                tokenizer.bos_token_id or tokenizer.eos_token_id,
                dtype=torch.long,
                device=images.device
            )
            
            generated_tokens = []
            
            for _ in range(max_length):
                # Get text embeddings
                text_embeds = self.language_model.transformer.wte(input_ids)
                
                # Combine with visual prefix
                combined_embeds = torch.cat([visual_prefix, text_embeds], dim=1)
                
                # Forward pass
                outputs = self.language_model(
                    inputs_embeds=combined_embeds,
                    return_dict=True
                )
                
                # Get next token logits
                next_token_logits = outputs.logits[:, -1, :] / temperature
                
                # Top-k sampling
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Check for EOS
                if (next_token == tokenizer.eos_token_id).all():
                    break
            
            # Decode
            captions = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
        
        return captions


# class ConceptualCaptionsDataset(Dataset):
#     """Load pre-downloaded images from disk"""
    
#     def __init__(self, cache_dir, tokenizer, config, split="train"):
#         self.cache_dir = Path(cache_dir)
#         self.tokenizer = tokenizer
#         self.config = config
        
#         # Load metadata
#         with open(self.cache_dir / "metadata.json") as f:
#             metadata = json.load(f)
        
#         self.captions = metadata['captions']
        
#         # Only include images that were successfully downloaded
#         self.valid_indices = [
#             i for i in range(len(self.captions))
#             if (self.cache_dir / f"{i:08d}.jpg").exists()
#         ]
        
#         print(f"Loaded {len(self.valid_indices)} cached images")
        
#         self.transform = transforms.Compose([
#             transforms.Resize((config.image_size, config.image_size)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
    
#     def __len__(self):
#         return len(self.valid_indices)
    
#     def __getitem__(self, idx):
#         real_idx = self.valid_indices[idx]
        
#         # Load image from disk
#         img_path = self.cache_dir / f"{real_idx:08d}.jpg"
#         image = Image.open(img_path).convert('RGB')
#         image = self.transform(image)
        
#         # Get caption and tokenize
#         caption = self.captions[real_idx]
#         encoding = self.tokenizer(
#             caption,
#             max_length=self.config.max_caption_length,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         )
        
#         input_ids = encoding['input_ids'].squeeze(0)
#         attention_mask = encoding['attention_mask'].squeeze(0)
#         labels = input_ids.clone()
#         labels[attention_mask == 0] = -100
        
#         return {
#             'images': image,
#             'input_ids': input_ids,
#             'attention_mask': attention_mask,
#             'labels': labels
#         }
    
class ConceptualCaptionsDataset(Dataset):
    def __init__(self, annotations_file, image_dir, tokenizer, config, split="train"):
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.config = config
        self.split = split

        # Load all annotations (even if images aren't downloaded yet)
        self.entries = []
        with open(annotations_file, 'r') as f:
            for line in f:
                self.entries.append(json.loads(line))

        print(f"Loaded {len(self.entries)} annotations for {split}")

        self.transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        # Loop until we find a valid, downloaded image
        # This handles race conditions where an image is listed but not yet fully present
        while True:
            entry = self.entries[idx]
            img_path = self.image_dir / entry['file_name']

            try:
                if not img_path.exists():
                    raise FileNotFoundError(f"Not found: {img_path}")

                # Try opening - this will fail if currently being written to
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
                
                caption = entry['text']
                encoding = self.tokenizer(
                    caption,
                    max_length=self.config.max_caption_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].squeeze(0)
                attention_mask = encoding['attention_mask'].squeeze(0)
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100
                
                return {
                    'images': image,
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                }

            except (FileNotFoundError, OSError, Image.UnidentifiedImageError):
                # If file is missing or corrupted (e.g., incomplete download), skip to next
                idx = (idx + 1) % len(self.entries)
                # A small sleep can help avoid hammering disk if many are missing
                # time.sleep(0.01)

def train_frozen(config: FrozenConfig):
    """Main training function"""
    
    # Setup
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb (optional)
    # wandb.init(project="frozen-replication", config=vars(config))
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config.language_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    print("Initializing Frozen model...")
    model = FrozenModel(config).to(config.device)
    
    # Initialize datasets
    # NOTE: Update these paths to your downloaded Conceptual Captions data
    DATA_ROOT = Path("conceptual_captions_data")

    train_dataset = ConceptualCaptionsDataset(
        annotations_file=str(DATA_ROOT / "train.jsonl"),
        image_dir=str(DATA_ROOT), # The jsonl now contains relative paths like "train/00001.jpg"
        tokenizer=tokenizer,
        config=config,
        split="train"
    )
    
    # Only enable this if you ran the downloader with split="validation"
    val_dataset = ConceptualCaptionsDataset(
       annotations_file=str(DATA_ROOT / "validation.jsonl"),
       image_dir=str(DATA_ROOT),
       tokenizer=tokenizer,
       config=config,
       split="val"
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Optimizer (only vision encoder parameters)
    optimizer = torch.optim.Adam(
        model.vision_encoder.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay
    )
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if config.fp16 else None
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for step, batch in enumerate(pbar):
            # Move to device
            images = batch['images'].to(config.device)
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)
            
            # Forward pass with mixed precision
            if config.fp16:
                with torch.amp.autocast('cuda'):
                    logits, loss = model(images, input_ids, attention_mask, labels)
                    loss = loss / config.gradient_accumulation_steps
            else:
                logits, loss = model(images, input_ids, attention_mask, labels)
                loss = loss / config.gradient_accumulation_steps
            
            # Backward pass
            if config.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (step + 1) % config.gradient_accumulation_steps == 0:
                if config.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
                global_step += 1
            
            # Logging
            epoch_loss += loss.item() * config.gradient_accumulation_steps
            pbar.set_postfix({'loss': loss.item() * config.gradient_accumulation_steps})
            
            # Validation
            if global_step % 1000 == 0:
                val_loss = validate(model, val_loader, config)
                print(f"\nStep {global_step} - Val Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, global_step, val_loss, config)
                
                model.train()
        
        # Epoch complete
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        
        # Validation
        val_loss = validate(model, val_loader, config)
        print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        save_checkpoint(model, optimizer, global_step, val_loss, config, epoch=epoch)
    
    print("Training complete!")


def validate(model: FrozenModel, val_loader: DataLoader, config: FrozenConfig) -> float:
    """Validation loop"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            images = batch['images'].to(config.device)
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)
            
            if config.fp16:
                with torch.cuda.amp.autocast():
                    _, loss = model(images, input_ids, attention_mask, labels)
            else:
                _, loss = model(images, input_ids, attention_mask, labels)
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def save_checkpoint(
    model: FrozenModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    val_loss: float,
    config: FrozenConfig,
    epoch: Optional[int] = None
):
    """Save model checkpoint"""
    checkpoint = {
        'step': step,
        'epoch': epoch,
        'vision_encoder_state_dict': model.vision_encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'config': vars(config)
    }
    
    filename = f"frozen_checkpoint_step{step}.pt" if epoch is None else f"frozen_checkpoint_epoch{epoch}.pt"
    path = Path(config.checkpoint_dir) / filename
    
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


if __name__ == "__main__":
    # Initialize configuration
    config = FrozenConfig()
    
    # For full-scale training (requires 40GB+ VRAM):
    # config.language_model_name = "EleutherAI/gpt-neo-2.7B"
    # config.batch_size = 4
    # config.gradient_accumulation_steps = 32
    
    # Start training
    train_frozen(config)