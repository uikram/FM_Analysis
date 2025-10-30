"""
Frozen Training Script
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
from frozen_model import FrozenModel

class DummyImageCaptionDataset(Dataset):
    """Dummy dataset for image captioning"""
    def __init__(self, size=1000, transform=None):
        self.size = size
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.captions = [
            "a dog playing in the park",
            "a cat sitting on a chair",
            "a bird flying in the sky"
        ]

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    print("Initializing Frozen model...")
    model = FrozenModel(
        lm_model_name='gpt2',
        num_visual_tokens=args.num_visual_tokens,
        freeze_lm=True
    )
    model = model.to(device)

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nParameter Efficiency:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable (vision encoder): {trainable_params:,}")
    print(f"  Frozen (LM): {total_params - trainable_params:,}")
    print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")

    # Datasets
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = DummyImageCaptionDataset(args.train_size, transform)
    val_dataset = DummyImageCaptionDataset(args.val_size, transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4)

    # Optimizer - only vision encoder
    optimizer = optim.AdamW(model.vision_encoder.parameters(), lr=args.lr, weight_decay=0.01)

    print("\nStarting training...")
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]'):
            images = batch['images'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs('../checkpoints', exist_ok=True)
            torch.save(model.vision_encoder.state_dict(), '../checkpoints/frozen_vision_encoder.pt')
            print(f'  ✓ Saved checkpoint (val_loss: {avg_val_loss:.4f})')

    print(f"\nTraining completed! Best val loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num_visual_tokens', type=int, default=2)
    parser.add_argument('--train_size', type=int, default=1000)
    parser.add_argument('--val_size', type=int, default=200)
    args = parser.parse_args()

    train_frozen(args)
