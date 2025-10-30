"""
LoRA Training Script
Standard training without GPU-specific optimizations
"""

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import argparse
import os

# Import from the lora_model.py file in the same directory
from lora_model import apply_lora_to_model

class DummyTextDataset(Dataset):
    """Dummy text classification dataset"""
    def __init__(self, size=1000, num_classes=5):
        self.size = size
        self.num_classes = num_classes
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.texts = [
            "This is great!", "I don't like this", "Neutral opinion",
            "Amazing experience", "Could be better"
        ]

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
    # Use cuda:0 if available, as set by CUDA_VISIBLE_DEVICES
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=args.num_classes
    )

    # Apply LoRA
    # Target distilbert's attention linear layers: q_lin and v_lin
    print(f"Applying LoRA (rank={args.rank}, alpha={args.alpha})...")
    model = apply_lora_to_model(model, rank=args.rank, alpha=args.alpha,
                                target_modules=['q_lin', 'v_lin'])
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nParameter Efficiency:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Reduction: {total_params / (trainable_params + 1e-9):.1f}x") # Avoid division by zero

    # Datasets
    train_dataset = DummyTextDataset(args.train_size, args.num_classes)
    val_dataset = DummyTextDataset(args.val_size, args.num_classes)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Optimizer - only optimize the LoRA parameters
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01)

    best_acc = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        print(f'Epoch {epoch+1}: Loss={avg_train_loss:.4f}, Acc={accuracy:.4f}')

        # Save the best model
        if accuracy > best_acc:
            best_acc = accuracy
            # Save checkpoint relative to this script's directory
            os.makedirs('../checkpoints', exist_ok=True)
            # Save only the LoRA parameters
            lora_state = {k: v for k, v in model.state_dict().items() if 'lora' in k.lower()}
            torch.save(lora_state, '../checkpoints/lora_best.pt')
            print(f'  ✓ Saved checkpoint to ../checkpoints/lora_best.pt')

    print(f"\nTraining completed! Best accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--alpha', type=int, default=1)
    parser.add_argument('--train_size', type=int, default=1000)
    parser.add_argument('--val_size', type=int, default=200)
    parser.add_argument('--num_classes', type=int, default=5)
    args = parser.parse_args()

    train_lora(args)
