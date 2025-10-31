"""
LoRA Training Script for Natural Language Inference
Adapts DistilBERT with LoRA for the MultiNLI task
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import time
from pathlib import Path

from lora_model import apply_lora_to_model

def get_nli_dataloaders(batch_size=32, max_length=128):
    """Load and preprocess MultiNLI dataset"""
    print("Loading MultiNLI dataset...")
    
    # Load dataset
    dataset = load_dataset("multi_nli")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    def tokenize_function(examples):
        # Combine premise and hypothesis
        return tokenizer(
            examples['premise'],
            examples['hypothesis'],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
    
    # Tokenize datasets
    train_dataset = dataset['train'].map(
        tokenize_function,
        batched=True,
        remove_columns=['premise', 'hypothesis', 'idx']
    )
    val_matched = dataset['validation_matched'].map(
        tokenize_function,
        batched=True,
        remove_columns=['premise', 'hypothesis', 'idx']
    )
    
    # Rename label column
    train_dataset = train_dataset.rename_column("label", "labels")
    val_matched = val_matched.rename_column("label", "labels")
    
    # Set format for PyTorch
    train_dataset.set_format("torch")
    val_matched.set_format("torch")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_matched,
        batch_size=batch_size,
        shuffle=False
    )
    
    print(f"✓ Loaded {len(train_dataset)} training samples")
    print(f"✓ Loaded {len(val_matched)} validation samples")
    
    return train_loader, val_loader train_loader, val_loader

def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    
    progress = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(progress):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == batch['labels']).sum().item()
        total += len(batch['labels'])
        
        # Update progress
        avg_loss = total_loss / (batch_idx + 1)
        accuracy = correct / total
        progress.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'acc': f'{accuracy*100:.1f}%'
        })
    
    epoch_time = time.time() - start_time
    print(f'Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {accuracy*100:.1f}%, '
          f'Time = {epoch_time:.1f}s')
    
    return avg_loss, accuracy

def evaluate(model, val_loader, device):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            
            total_loss += outputs.loss.item()
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == batch['labels']).sum().item()
            total += len(batch['labels'])
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def train_lora(
    rank=8,
    alpha=16,
    epochs=10,
    batch_size=32,
    learning_rate=1e-3,
    device='cuda',
    save_dir='../checkpoints'
):
    """Train LoRA adapter for NLI task"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load base model
    print("\nLoading base model...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=3  # entailment, neutral, contradiction
    )
    
    # Apply LoRA
    print(f"Applying LoRA (rank={rank}, alpha={alpha})...")
    model = apply_lora_to_model(
        base_model,
        rank=rank,
        alpha=alpha,
        target_modules=['q_lin', 'v_lin']
    )
    
    # Print parameter stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameter stats:")
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Trainable %:      {(trainable_params/total_params)*100:.2f}%")
    
    model = model.to(device)
    
    # Setup training
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate
    )
    train_loader, val_loader = get_nli_dataloaders(batch_size=batch_size)
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_val_acc = 0
    best_epoch = 0
    
    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, epoch
        )
        
        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"Validation: Loss = {val_loss:.4f}, Accuracy = {val_acc*100:.1f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                model.state_dict(),
                save_dir / 'lora_nli_best.pt'
            )
            print(f"✓ Saved new best model (accuracy: {best_val_acc*100:.1f}%)")
    
    print(f"\nTraining complete! Best model from epoch {best_epoch}")
    print(f"Best validation accuracy: {best_val_acc*100:.1f}%")
    print(f"Model saved to {save_dir / 'lora_nli_best.pt'}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train LoRA adapter for MultiNLI')
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--alpha', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-dir', default='../checkpoints')
    args = parser.parse_args()
    
    train_lora(
        rank=args.rank,
        alpha=args.alpha,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        save_dir=args.save_dir
    )