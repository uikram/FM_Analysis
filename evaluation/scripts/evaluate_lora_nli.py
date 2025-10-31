"""
LoRA Evaluation on Natural Language Inference
Tests LoRA adaptation on MultiNLI classification task
"""

import os
import sys
import json
import torch
import argparse
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# Import LoRA utilities
from lora_model import apply_lora_to_model

def get_nli_dataloaders(batch_size=32, max_length=128):
    """Load and preprocess MultiNLI dataset"""
    print("Loading MultiNLI dataset...")
    
    # Load dataset (use matched validation set for evaluation)
    dataset = load_dataset("multi_nli")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    def tokenize_function(examples):
        # Combine premise and hypothesis with separator
        text_pairs = list(zip(examples['premise'], examples['hypothesis']))
        tokenized = tokenizer(
            text_pairs,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        return tokenized
    
    # Tokenize validation set
    val_matched = dataset['validation_matched']
    tokenized_val = val_matched.map(
        tokenize_function, 
        batched=True, 
        remove_columns=val_matched.column_names
    )
    tokenized_val = tokenized_val.rename_column("label", "labels")
    tokenized_val.set_format("torch")
    
    # Create dataloader
    val_loader = DataLoader(
        tokenized_val, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    print(f"✓ Loaded {len(tokenized_val)} validation samples")
    return val_loader

def evaluate_lora_nli(model_path, rank=4, alpha=1, batch_size=32, device='cuda'):
    """Evaluate LoRA on MultiNLI task"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load base model (3 classes for entailment/neutral/contradiction)
    print("\nLoading base model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', 
        num_labels=3
    )
    
    # Apply LoRA
    print(f"Applying LoRA (rank={rank}, alpha={alpha})...")
    model = apply_lora_to_model(
        model, 
        rank=rank, 
        alpha=alpha,
        target_modules=['q_lin', 'v_lin']
    )
    
    # Get parameter stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Load trained weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        print(f"Error: Model checkpoint not found at {model_path}")
        sys.exit(1)
        
    model = model.to(device)
    model.eval()
    
    # Load validation data
    val_loader = get_nli_dataloaders(batch_size=batch_size)
    
    # Evaluate
    print("\nEvaluating...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            
            preds = outputs.logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, 
        all_preds,
        target_names=['Contradiction', 'Neutral', 'Entailment'],
        output_dict=True
    )
    
    # Prepare results
    results = {
        'task': 'nli',
        'accuracy': float(accuracy),
        'classification_report': report,
        'parameter_stats': {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'trainable_pct': (trainable_params / total_params) * 100
        }
    }
    
    # Print summary
    print(f"\n{'='*70}\nLoRA NLI EVALUATION RESULTS\n{'='*70}")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"\nParameter Stats:")
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Trainable %:      {(trainable_params/total_params)*100:.2f}%")
    
    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'lora')
    os.makedirs(results_dir, exist_ok=True)
    
    results_path = os.path.join(results_dir, 'nli_evaluation.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\n✓ Results saved to {results_path}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate LoRA on MultiNLI')
    parser.add_argument('--model_path', required=True, help='Path to trained LoRA model')
    parser.add_argument('--rank', type=int, default=4, help='LoRA rank')
    parser.add_argument('--alpha', type=float, default=1, help='LoRA alpha scaling')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', default='cuda', help='Device to run on (cuda/cpu)')
    args = parser.parse_args()
    
    evaluate_lora_nli(
        args.model_path,
        args.rank,
        args.alpha,
        args.batch_size,
        args.device
    )