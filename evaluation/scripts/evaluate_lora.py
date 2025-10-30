"""
LoRA Evaluation Script
Evaluates LoRA fine-tuning on IMDB sentiment classification

Usage:
  cd evaluation/scripts
  python evaluate_lora.py --model_path ../../LoRA_2106.09685v1/checkpoints/lora_best.pt
"""
import sys
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import json
import time

# Import from installed local package
from lora_model import apply_lora_to_model

# Import Hugging Face datasets
from datasets import load_dataset

def get_lora_dataloaders(batch_size=32, max_length=256):
    """Loads and tokenizes IMDB dataset from Hugging Face"""
    print("Loading IMDB dataset...")
    
    # This will download and cache the dataset
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)
        
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    
    test_dataset = tokenized_datasets["test"]
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✓ Loaded {len(test_dataset)} test samples")
    return test_loader

def evaluate_lora_sentiment(model_path, rank=4, alpha=1, batch_size=32, device='cuda'):
    """Evaluate LoRA on IMDB sentiment classification"""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load base model
    print("\nLoading base model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=2
    )
    
    # Apply LoRA
    print(f"Applying LoRA (rank={rank}, alpha={alpha})...")
    model = apply_lora_to_model(
        model, rank=rank, alpha=alpha, target_modules=['q_lin', 'v_lin']
    )
    
    # Get parameter stats *before* loading checkpoint
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Load LoRA weights
    if os.path.exists(model_path):
        lora_state = torch.load(model_path, map_location=device)
        model.load_state_dict(lora_state, strict=False)
        print("✓ Loaded LoRA checkpoint")
    else:
        print(f"⚠ No checkpoint found at {model_path}, evaluating with random LoRA weights")
    
    model = model.to(device)
    model.eval()
    
    # Load dataset
    test_loader = get_lora_dataloaders(batch_size=batch_size)
    
    # Evaluate
    print("\nEvaluating...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, target_names=['Negative', 'Positive'], output_dict=True
    )
    
    results = {
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': report,
        'parameter_stats': {
            'total_params': int(total_params),
            'trainable_params': int(trainable_params),
            'trainable_percentage': float(100 * trainable_params / total_params),
            'reduction_factor': float(total_params / (trainable_params + 1e-9))
        }
    }
    
    # Print results
    print(f"\n{'='*70}\nLoRA SENTIMENT CLASSIFICATION RESULTS (IMDB)\n{'='*70}")
    print(f"  Accuracy:  {accuracy*100:.2f}%")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'lora')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_dir}/")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate LoRA on IMDB')
    parser.add_argument('--model_path', type=str,
                       default='../../LoRA_2106.09685v1/checkpoints/lora_best.pt',
                       help='Path to LoRA checkpoint')
    
    # ADD THESE TWO LINES
    parser.add_argument('--rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--alpha', type=int, default=1, help='LoRA alpha scaling')
    
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2' # Not needed, you set this in your command
    
    # UPDATE THIS FUNCTION CALL TO PASS THE NEW ARGS
    evaluate_lora_sentiment(
        args.model_path, args.rank, args.alpha, args.batch_size, args.device
    )