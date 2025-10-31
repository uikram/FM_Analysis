"""
CLIP Training Script - Optimized for RTX A5000 (24GB VRAM)
*** MODIFIED TO USE REAL DATA (Flickr8k) ***

Run with: CUDA_VISIBLE_DEVICES=2 python train.py
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2' # Better to set in terminal
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from PIL import Image
from tqdm import tqdm
import numpy as np
import argparse
from clip_model import CLIP, contrastive_loss
from datasets import load_dataset # <-- ADDED THIS

# A5000 optimized settings
BATCH_SIZE = 64
NUM_WORKERS = 4
MIXED_PRECISION = True

class CombinedImageCaptionDataset(Dataset):
    """
    Dataset that combines multiple image-caption datasets for CLIP training.
    """
    def __init__(self, datasets, tokenizer, transform):
        self.datasets = datasets
        self.dataset_sizes = [len(ds) for ds in datasets]
        self.total_size = sum(self.dataset_sizes)
        self.tokenizer = tokenizer
        self.transform = transform
        
        # Calculate cumulative sizes for dataset indexing
        self.cumsum_sizes = np.cumsum([0] + self.dataset_sizes)

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        # Find which dataset this index belongs to
        dataset_idx = np.searchsorted(self.cumsum_sizes[1:], idx, side='right')
        local_idx = idx - self.cumsum_sizes[dataset_idx]
        
        # Get item from appropriate dataset
        item = self.datasets[dataset_idx][local_idx]
        
        # Load and transform the image
        image = item['image']
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = self.transform(image)
        
        # Get caption (handle different dataset formats)
        caption = item['captions'][0] if isinstance(item.get('captions'), list) else item['caption']
        
        # Tokenize caption
        encoded = self.tokenizer(
            caption,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        
        return image, encoded['input_ids'].squeeze(0)

class Flickr8kDataset(Dataset):
    """
    Dataset for loading Flickr8k for CLIP training.
    This will load an image and one of its corresponding captions.
    """
    def __init__(self, hf_dataset, tokenizer, transform):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        
        # Load and transform the image
        image = item['image']
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = self.transform(image)
        
        # Get the first caption (for contrastive learning, we just need one)
        caption = item['captions'][0]
        
        # Tokenize the caption
        tokens = self.tokenizer(caption, padding='max_length', truncation=True,
                               max_length=77, return_tensors='pt')

        return {
            'image': image,
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0)
        }

def train_clip(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model = CLIP(embed_dim=512).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --- LOAD COMBINED DATASETS ---
    print("Loading datasets...")
    
    # Load Flickr8k
    print("1. Loading Flickr8k...")
    flickr8k = load_dataset("tsystems/flickr8k")
    if 'validation' not in flickr8k:
        train_val_split = flickr8k['train'].train_test_split(test_size=0.1, seed=42)
        flickr8k['train'] = train_val_split['train']
        flickr8k['validation'] = train_val_split['test']
    
    # Load MS-COCO
    print("\n2. Loading MS-COCO captions...")
    coco = load_dataset("coco_captions")
    
    # Create combined training dataset
    train_datasets = [
        flickr8k['train'],
        coco['train']
    ]
    val_datasets = [
        flickr8k['validation'],
        coco['validation']
    ]
    
    train_dataset = CombinedImageCaptionDataset(
        train_datasets,
        tokenizer=tokenizer,
        transform=transform
    )
    
    val_dataset = CombinedImageCaptionDataset(
        val_datasets,
        tokenizer=tokenizer,
        transform=transform
    )
    
    print(f"\n✓ Loaded combined datasets:")
    print(f"  Training samples: {len(train_dataset):,}")
    print(f"  Validation samples: {len(val_dataset):,}")
    # --- END OF DATASET LOADING ---

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda') if MIXED_PRECISION and torch.cuda.is_available() else None

    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}'):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast('cuda'):
                    logits_i, logits_t = model(images, input_ids, attention_mask)
                    loss = contrastive_loss(logits_i, logits_t)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits_i, logits_t = model(images, input_ids, attention_mask)
                loss = contrastive_loss(logits_i, logits_t)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                with torch.amp.autocast('cuda') if scaler else torch.no_grad():
                    logits_i, logits_t = model(images, input_ids, attention_mask)
                    loss = contrastive_loss(logits_i, logits_t)
                
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch+1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}')

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            os.makedirs('../checkpoints', exist_ok=True)
            # Save the model state dict
            torch.save(model.state_dict(), '../checkpoints/clip_best.pt')
            print(f'  ✓ Saved checkpoint (val_loss: {avg_val_loss:.4f})')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=1e-4) # You might need to tune this
    # The dummy dataset was tiny. You can remove these args if you want to use the full dataset
    # parser.add_argument('--train_size', type=int, default=1000)
    # parser.add_argument('--val_size', type=int, default=200)
    args = parser.parse_args()

    train_clip(args)