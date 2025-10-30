"""
LoRA Training Script - Optimized for RTX A5000

This script is intended to be run by setting the GPU in the terminal:
CUDA_VISIBLE_DEVICES=2 python src/train_a5000.py
"""
import os

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import argparse

from lora_model import apply_lora_to_model

# Tuned for A5000
BATCH_SIZE = 32 
NUM_WORKERS = 4 # Use more workers for data loading

class DummyTextDataset(Dataset):
    """Dummy text classification dataset"""
    def __init__(self, size=1000, num_classes=5):
        self.size = size
        self.num_classes = num_classes
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.texts = ["Great!", "Bad", "Okay", "Amazing", "Poor"]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        text = np.random.choice(self.texts)
        label = np.random.randint(0, self.num_classes)
        encoding = self.tokenizer(text, padding='max_length', truncation=True,
                                  max_length=128, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label)
        }

def train_lora(args):
    # As per the CUDA_VISIBLE_DEVICES logic, the selected GPU (e.g., #2)
    # will be seen by torch as 'cuda:0'.
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print(f"--- A5000 Optimized Training ---")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        # This will print the name of the GPU you selected
        # (e.g., "NVIDIA RTX A5000")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}") 

    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=args.num_classes)
    # Target distilbert's attention linear layers
    model = apply_lora_to_model(model, rank=args.rank, alpha=args.alpha, target_modules=['q_lin', 'v_lin'])
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total: {total_params:,}, Trainable: {trainable_params:,}, Reduction: {total_params/(trainable_params+1e-9):.1f}x")

    train_dataset = DummyTextDataset(args.train_size, args.num_classes)
    val_dataset = DummyTextDataset(args.val_size, args.num_classes)

    # Use optimized DataLoader settings
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True)

    lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01)

    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            # Data is already on pinned memory, .to(device) should be faster
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        print(f'Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Acc={acc:.4f}')

        if acc > best_acc:
            best_acc = acc
            os.makedirs('../checkpoints', exist_ok=True)
            lora_state = {k: v for k, v in model.state_dict().items() if 'lora' in k.lower()}
            torch.save(lora_state, '../checkpoints/lora_a5000_best.pt')
            print(f'  ✓ Saved checkpoint to ../checkpoints/lora_a5000_best.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--rank', type=int, default=8) # Default rank 8 for A5000
    parser.add_argument('--alpha', type=int, default=1)
    parser.add_argument('--train_size', type=int, default=5000) # Larger dataset
    parser.add_argument('--val_size', type=int, default=1000)
    parser.add_argument('--num_classes', type=int, default=5)
    args = parser.parse_args()
    train_lora(args)
