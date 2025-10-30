"""
CLIP Evaluation Script
Evaluates CLIP on CIFAR-10 zero-shot image classification

Usage:
  cd evaluation/scripts
  python evaluate_clip.py --model_path ../../CLIP_radford21a/checkpoints/clip_best.pt
"""
import sys
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import json
from transformers import AutoTokenizer

# No more sys.path hacks! We can import directly
# because we installed the package with pip install -e
from clip_model import CLIP

def get_clip_dataloaders(batch_size, root='../data'):
    """Loads CIFAR-10, downloading if necessary"""
    os.makedirs(root, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    class_names = test_dataset.classes
    return test_loader, class_names

def evaluate_clip_zero_shot(model_path, batch_size=64, device='cuda'):
    """Evaluate CLIP on CIFAR-10 zero-shot classification"""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading CLIP model...")
    model = CLIP(embed_dim=512)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        # Handle both state_dict and full model saves
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✓ Loaded trained model checkpoint")
    else:
        print(f"⚠ No checkpoint found at {model_path}, evaluating untrained model")
    
    model = model.to(device)
    model.eval()
    
    # Load dataset
    print("\nLoading CIFAR-10 dataset...")
    test_loader, class_names = get_clip_dataloaders(batch_size=batch_size)
    print(f"✓ Loaded {len(test_loader.dataset)} test images")
    print(f"Classes: {class_names}")
    
    # Prepare text prompts
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    templates = ["a photo of a {}", "a picture of a {}", "an image of a {}"]
    
    all_text_prompts = []
    for template in templates:
        for name in class_names:
            all_text_prompts.append(template.format(name))
    
    text_tokens = tokenizer(
        all_text_prompts, padding=True, truncation=True, max_length=77, return_tensors='pt'
    ).to(device)
    
    # Encode text descriptions once
    print("\nEncoding text descriptions...")
    with torch.no_grad():
        text_features = model.encode_text(
            text_tokens['input_ids'], text_tokens['attention_mask']
        )
        text_features = text_features.view(len(templates), len(class_names), -1)
        text_features = text_features.mean(dim=0)  # [num_classes, embed_dim]
    
    # Evaluate
    print("\nEvaluating on test set...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            images = batch[0].to(device)
            labels = batch[1]
            
            image_features = model.encode_image(images)
            similarity = image_features @ text_features.T
            predictions = similarity.argmax(dim=1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True
    )
    
    results = {
        'accuracy': float(accuracy),
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': report
    }
    
    # Print and save results
    print(f"\n{'='*70}\nCLIP ZERO-SHOT EVALUATION RESULTS (CIFAR-10)\n{'='*70}")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'clip')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_dir}/")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate CLIP on CIFAR-10')
    parser.add_argument('--model_path', type=str, 
                       default='../../CLIP_radford21a/checkpoints/clip_best.pt',
                       help='Path to trained model checkpoint')
    args = parser.parse_args()
    
    evaluate_clip_zero_shot(args.model_path)