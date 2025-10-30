"""
Frozen Training Script - Optimized for RTX A5000
Run with: CUDA_VISIBLE_DEVICES=2 python train_a5000.py
"""
import os
# This line is risky, recommend setting in terminal instead.
# os.environ['CUDA_VISIBLE_DEVICES'] = '2' 
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

BATCH_SIZE = 16
NUM_WORKERS = 4

class DummyImageCaptionDataset(Dataset):
    def __init__(self, size=1000, transform=None):
        self.size = size
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.captions = ["a dog playing", "a cat sitting", "a bird flying"]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        if self.transform:
            img = self.transform(img)

        caption = np.random.choice(self.captions)
        tokens = self.tokenizer(caption, padding='max_length', truncation=True,
                               max_length=50, return_tensors='pt')

        return {
            'images': img,
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'labels': tokens['input_ids'].squeeze(0)
        }

def train_frozen(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = FrozenModel('gpt2', args.num_visual_tokens, freeze_lm=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total: {total_params:,}, Trainable: {trainable_params:,}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = DummyImageCaptionDataset(args.train_size, transform)
    val_dataset = DummyImageCaptionDataset(args.val_size, transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

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
            torch.save(model.vision_encoder.state_dict(), '../checkpoints/frozen_vision_encoder.pt')
            print(f'  ✓ Saved checkpoint (val_loss: {avg_val:.4f})')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_visual_tokens', type=int, default=2)
    parser.add_argument('--train_size', type=int, default=1000)
    parser.add_argument('--val_size', type=int, default=200)
    args = parser.parse_args()
    train_frozen(args)
