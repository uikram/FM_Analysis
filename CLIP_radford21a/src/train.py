"""
CLIP Training Script
Standard training without GPU-specific optimizations
"""
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from PIL import Image
from tqdm import tqdm
import numpy as np
import argparse
import os
from clip_model import CLIP, contrastive_loss

class DummyDataset(Dataset):
    """Dummy dataset for demonstration"""
    def __init__(self, size=1000, transform=None):
        self.size = size
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.captions = [
            "a dog playing in the park",
            "a cat sitting on a chair",
            "a bird flying in the sky",
            "a car driving on the road",
            "a person walking on the street"
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = CLIP(embed_dim=512, temperature=0.07).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = DummyDataset(args.train_size, transform)
    val_dataset = DummyDataset(args.val_size, transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]'):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()
            logits_per_image, logits_per_text = model(images, input_ids, attention_mask)
            loss = contrastive_loss(logits_per_image, logits_per_text)
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

                logits_per_image, logits_per_text = model(images, input_ids, attention_mask)
                loss = contrastive_loss(logits_per_image, logits_per_text)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs('../checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, '../checkpoints/clip_best.pt')
            print(f'  ✓ Saved checkpoint (val_loss: {avg_val_loss:.4f})')

        scheduler.step()

    print(f"\nTraining completed! Best val loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train_size', type=int, default=1000)
    parser.add_argument('--val_size', type=int, default=200)
    args = parser.parse_args()

    train_clip(args)
