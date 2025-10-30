"""
Frozen Training Script - Optimized for RTX A5000
*** MODIFIED TO USE REAL DATA (Flickr8k) ***

Run with: CUDA_VISIBLE_DEVICES=2 python train.py
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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
from frozen_model import FrozenModel
from datasets import load_dataset # <-- ADDED THIS

BATCH_SIZE = 16
NUM_WORKERS = 4

class Flickr8kDataset(Dataset):
    """
    Dataset for loading Flickr8k for Frozen model training.
    """
    def __init__(self, hf_dataset, tokenizer, transform, max_length=50):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        
        # Load and transform the image
        image = item['image']
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = self.transform(image)
        
        # Get the first caption
        caption = item['captions'][0]
        
        # Tokenize the caption
        tokens = self.tokenizer(caption, padding='max_length', truncation=True,
                               max_length=self.max_length, return_tensors='pt')
        
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)

        return {
            'images': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone() # For language modeling, labels are the same as input_ids
        }

def train_frozen(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- TOKENIZER ---
    # We set the pad token to the eos token
    # This is the source of the warning, but it's standard for GPT-2 generation
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    # --- END TOKENIZER ---

    model = FrozenModel('gpt2', args.num_visual_tokens, freeze_lm=True).to(device)
    
    # If the tokenizer was resized (e.g., new pad token), we'd resize model embeddings here
    # But since we're using eos_token, it's fine.

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total: {total_params:,}, Trainable: {trainable_params:,}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --- LOAD REAL DATASET ---
    print("Loading Flickr8k dataset...")
    hf_dataset = load_dataset("tsystems/flickr8k")
    
    # Create a 90/10 train/validation split
    if 'validation' not in hf_dataset:
        print("Creating validation split...")
        train_val_split = hf_dataset['train'].train_test_split(test_size=0.1, seed=42)
        hf_dataset['train'] = train_val_split['train']
        hf_dataset['validation'] = train_val_split['test']
        
    train_dataset = Flickr8kDataset(
        hf_dataset['train'], 
        tokenizer=tokenizer, 
        transform=transform
    )
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

    # We only optimize the vision encoder parameters, as the LM is frozen
    optimizer = optim.AdamW(model.vision_encoder.parameters(), lr=args.lr, weight_decay=0.01)

    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            images = batch['images'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            
            # Forward pass through the FrozenModel
            outputs = model(images=images, input_ids=input_ids,
                          attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(images=images, input_ids=input_ids,
                              attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f'Epoch {epoch+1}: Train={avg_train:.4f}, Val={avg_val:.4f}')

        if avg_val < best_loss:
            best_loss = avg_val
            os.makedirs('../checkpoints', exist_ok=True)
            # Save only the vision encoder weights, as that's all we trained
            torch.save(model.vision_encoder.state_dict(), '../checkpoints/frozen_vision_encoder.pt')
            print(f'  ✓ Saved checkpoint (val_loss: {avg_val:.4f})')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_visual_tokens', type=int, default=2)
    # The dummy dataset args are no longer needed
    # parser.add_argument('--train_size', type=int, default=1000)
    # parser.add_argument('--val_size', type=int, default=200)
    args = parser.parse_args()
    
    train_frozen(args)