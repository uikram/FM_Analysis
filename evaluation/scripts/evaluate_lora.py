"""
LoRA Evaluation Script  
Evaluates LoRA fine-tuning on IMDB sentiment classification
"""

import sys
sys.path.append('../LoRA_2106.09685v1/src')
sys.path.append('../evaluation/datasets')

import torch
from lora_model import apply_lora_to_model
from lora_datasets import get_lora_dataloaders
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import json
import os
import time

def evaluate_lora_sentiment(model_path, rank=4, alpha=1, batch_size=32, device='cuda'):
    """Evaluate LoRA on IMDB sentiment classification"""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
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
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\nParameter Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable (LoRA): {trainable_params:,}")
    print(f"  Frozen (base): {frozen_params:,}")
    print(f"  Reduction factor: {total_params / trainable_params:.1f}x")
    print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    # Load LoRA weights if available
    if os.path.exists(model_path):
        lora_state = torch.load(model_path, map_location=device)
        model.load_state_dict(lora_state, strict=False)
        print("✓ Loaded LoRA checkpoint")
    else:
        print("⚠ No checkpoint found, evaluating with random LoRA weights")
    
    model = model.to(device)
    model.eval()
    
    # Load dataset
    print("\nLoading IMDB dataset...")
    _, test_loader = get_lora_dataloaders(batch_size=batch_size)
    print(f"✓ Loaded {len(test_loader.dataset)} test samples")
    
    # Evaluate
    print("\nEvaluating...")
    all_preds = []
    all_labels = []
    all_probs = []
    inference_times = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            # Measure inference time
            start_time = time.time()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            predictions = torch.argmax(outputs.logits, dim=1)
            probs = torch.softmax(outputs.logits, dim=1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    report = classification_report(
        all_labels, all_preds, 
        target_names=['Negative', 'Positive'],
        output_dict=True
    )
    
    # Performance metrics
    avg_inference_time = sum(inference_times) / len(inference_times)
    samples_per_second = batch_size / avg_inference_time
    
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
            'frozen_params': int(frozen_params),
            'reduction_factor': float(total_params / trainable_params),
            'trainable_percentage': float(100 * trainable_params / total_params)
        },
        'performance': {
            'avg_inference_time_per_batch': float(avg_inference_time),
            'samples_per_second': float(samples_per_second)
        }
    }
    
    # Print results
    print(f"\n{'='*70}")
    print("LoRA SENTIMENT CLASSIFICATION RESULTS")
    print(f"{'='*70}")
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy*100:.2f}%")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  {'':12s} Pred Neg  Pred Pos")
    print(f"  True Neg:  {conf_matrix[0,0]:8d}  {conf_matrix[0,1]:8d}")
    print(f"  True Pos:  {conf_matrix[1,0]:8d}  {conf_matrix[1,1]:8d}")
    
    print(f"\nPer-Class Metrics:")
    for label in ['Negative', 'Positive']:
        metrics = report[label]
        print(f"  {label:10s}: "
              f"Precision={metrics['precision']:.3f}, "
              f"Recall={metrics['recall']:.3f}, "
              f"F1={metrics['f1-score']:.3f}")
    
    print(f"\nParameter Efficiency:")
    print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"  Reduction: {total_params/trainable_params:.0f}x fewer parameters to train")
    
    print(f"\nInference Performance:")
    print(f"  Throughput: {samples_per_second:.1f} samples/second")
    
    # Save results
    os.makedirs('../evaluation/results/lora', exist_ok=True)
    with open('../evaluation/results/lora/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    import numpy as np
    np.save('../evaluation/results/lora/confusion_matrix.npy', conf_matrix)
    
    print(f"\n✓ Results saved to evaluation/results/lora/")
    print(f"  - evaluation_results.json")
    print(f"  - confusion_matrix.npy")
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate LoRA on IMDB')
    parser.add_argument('--model_path', type=str,
                       default='../LoRA_2106.09685v1/checkpoints/lora_best.pt',
                       help='Path to LoRA checkpoint')
    parser.add_argument('--rank', type=int, default=4,
                       help='LoRA rank')
    parser.add_argument('--alpha', type=int, default=1,
                       help='LoRA alpha scaling')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    evaluate_lora_sentiment(
        args.model_path, args.rank, args.alpha, args.batch_size, args.device
    )
