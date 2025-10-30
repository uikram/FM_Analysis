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

    # --- LOAD REAL DATASET ---
    print("Loading Flickr8k dataset...")
    hf_dataset = load_dataset("tsystems/flickr8k")

    # Check if 'validation' split exists. If not, create it from 'train'.
    if 'validation' not in hf_dataset:
        print("No 'validation' split found. Creating one from the 'train' split (90% train / 10% val)...")
        # Split the 'train' dataset into 90% train and 10% validation
        train_val_split = hf_dataset['train'].train_test_split(test_size=0.1, seed=42)
        hf_dataset['train'] = train_val_split['train']
        hf_dataset['validation'] = train_val_split['test'] # Use the 'test' part as our validation

    train_dataset = Flickr8kDataset(
        hf_dataset['train'], 
        tokenizer=tokenizer, 
        transform=transform
    )
    # This will now work
    val_dataset = Flickr8kDataset(
        hf_dataset['validation'], 
        tokenizer=tokenizer, 
        transform=transform
    )
    print(f"✓ Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")
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