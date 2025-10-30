"""
CLIP Training Script - Optimized for RTX A5000 (24GB VRAM)
Run with: CUDA_VISIBLE_DEVICES=2 python train_a5000.py
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

# A5000 optimized settings
BATCH_SIZE = 64
NUM_WORKERS = 4
MIXED_PRECISION = True

class DummyDataset(Dataset):
    def __init__(self, size=1000, transform=None):
        self.size = size
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.captions = [
            "a dog playing", "a cat sitting", "a bird flying",
            "a car driving", "a person walking"
        ]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        if self.transform:
            img = self.transform(img)

        caption = np.random.choice(self.captions)
        tokens = self.tokenizer(caption, padding='max_length', truncation=True,
                               max_length=77, return_tensors='pt')

        return {
            'image': img,
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

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = DummyDataset(args.train_size, transform)
    val_dataset = DummyDataset(args.val_size, transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # scaler = torch.cuda.amp.GradScaler() if MIXED_PRECISION and torch.cuda.is_available() else None
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
                # with torch.cuda.amp.autocast():
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
                logits_i, logits_t = model(images, input_ids, attention_mask)
                loss = contrastive_loss(logits_i, logits_t)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch+1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}')

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            os.makedirs('../checkpoints', exist_ok=True)
            torch.save(model.state_dict(), '../checkpoints/clip_best.pt')
            print(f'  ✓ Saved checkpoint')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train_size', type=int, default=1000)
    parser.add_argument('--val_size', type=int, default=200)
    args = parser.parse_args()

    train_clip(args)
