"""
CLIP Evaluation Script
Evaluates CLIP on CIFAR-10 zero-shot image classification
"""

import sys
sys.path.append('../CLIP_radford21a/src')
sys.path.append('../evaluation/datasets')

import torch
from clip_model import CLIP
from transformers import AutoTokenizer
from clip_datasets import get_clip_dataloaders
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import json
import os

def evaluate_clip_zero_shot(model_path, batch_size=64, device='cuda'):
    """Evaluate CLIP on CIFAR-10 zero-shot classification"""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    print("\nLoading CLIP model...")
    model = CLIP(embed_dim=512)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✓ Loaded trained model checkpoint")
    else:
        print("⚠ No checkpoint found, evaluating untrained model")
    
    model = model.to(device)
    model.eval()
    
    # Load dataset
    print("\nLoading CIFAR-10 dataset...")
    _, test_loader, class_names = get_clip_dataloaders(
        batch_size=batch_size, 
        root='./evaluation/data'
    )
    print(f"✓ Loaded {len(test_loader.dataset)} test images")
    print(f"Classes: {class_names}")
    
    # Prepare text prompts for zero-shot
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Use multiple prompt templates for better performance
    templates = [
        "a photo of a {}",
        "a picture of a {}",
        "an image of a {}",
    ]
    
    all_text_prompts = []
    for template in templates:
        for name in class_names:
            all_text_prompts.append(template.format(name))
    
    # Tokenize all prompts
    text_tokens = tokenizer(
        all_text_prompts,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors='pt'
    ).to(device)
    
    # Encode text descriptions once
    print("\nEncoding text descriptions...")
    with torch.no_grad():
        text_features = model.encode_text(
            text_tokens['input_ids'],
            text_tokens['attention_mask']
        )
        
        # Average over templates
        text_features = text_features.view(len(templates), len(class_names), -1)
        text_features = text_features.mean(dim=0)  # [num_classes, embed_dim]
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    all_preds = []
    all_labels = []
    all_similarities = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            images = batch['image'].to(device)
            labels = batch['label']
            
            # Encode images
            image_features = model.encode_image(images)
            
            # Compute similarity with all classes
            similarity = image_features @ text_features.T
            predictions = similarity.argmax(dim=1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_similarities.append(similarity.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Per-class accuracy
    per_class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    
    # Top-5 accuracy
    all_similarities = np.concatenate(all_similarities, axis=0)
    top5_preds = np.argsort(all_similarities, axis=1)[:, -5:]
    top5_correct = sum(label in top5_preds[i] for i, label in enumerate(all_labels))
    top5_accuracy = top5_correct / len(all_labels)
    
    # Classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True
    )
    
    results = {
        'accuracy': float(accuracy),
        'top5_accuracy': float(top5_accuracy),
        'per_class_accuracy': {
            class_names[i]: float(per_class_acc[i]) 
            for i in range(len(class_names))
        },
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': report
    }
    
    # Print results
    print(f"\n{'='*70}")
    print("CLIP ZERO-SHOT EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"\nOverall Metrics:")
    print(f"  Top-1 Accuracy: {accuracy*100:.2f}%")
    print(f"  Top-5 Accuracy: {top5_accuracy*100:.2f}%")
    
    print(f"\nPer-Class Accuracy:")
    for name, acc in sorted(results['per_class_accuracy'].items(), 
                           key=lambda x: x[1], reverse=True):
        print(f"  {name:15s}: {acc*100:.2f}%")
    
    print(f"\nClassification Report:")
    for class_name in class_names:
        metrics = report[class_name]
        print(f"  {class_name:15s}: "
              f"Precision={metrics['precision']:.3f}, "
              f"Recall={metrics['recall']:.3f}, "
              f"F1={metrics['f1-score']:.3f}")
    
    # Save results
    os.makedirs('../evaluation/results/clip', exist_ok=True)
    with open('../evaluation/results/clip/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save confusion matrix
    np.save('../evaluation/results/clip/confusion_matrix.npy', conf_matrix)
    
    print(f"\n✓ Results saved to evaluation/results/clip/")
    print(f"  - evaluation_results.json")
    print(f"  - confusion_matrix.npy")
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate CLIP on CIFAR-10')
    parser.add_argument('--model_path', type=str, 
                       default='../CLIP_radford21a/checkpoints/clip_best.pt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    evaluate_clip_zero_shot(args.model_path, args.batch_size, args.device)
