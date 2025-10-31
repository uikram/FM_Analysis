"""
Enhanced CLIP Evaluation Script with Backbone Comparison
Compares ResNet-50 vs ViT backbones on zero-shot transfer
"""

import os
import sys
import json
import torch
import argparse
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Import CLIP model
from clip_model import CLIP

def evaluate_clip_backbone(model_path, backbone='resnet50', datasets=['cifar10'], batch_size=64, device='cuda'):
    """Evaluate CLIP with specific backbone on multiple datasets"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading CLIP model with {backbone} backbone...")
    model = CLIP(embed_dim=512, backbone=backbone)
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
    else:
        print(f"Error: Model checkpoint not found at {model_path}")
        sys.exit(1)
        
    model = model.to(device)
    model.eval()
    
    # Track results for each dataset
    results = {
        'backbone': backbone,
        'datasets': {}
    }
    
    # Standard transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    for dataset_name in datasets:
        print(f"\nEvaluating on {dataset_name}...")
        
        # Load dataset
        if dataset_name == 'cifar10':
            dataset = load_dataset('cifar10', split='test')
        elif dataset_name == 'food101':
            dataset = load_dataset('food101', split='validation[:2000]')
        elif dataset_name == 'stl10':
            dataset = load_dataset('stl10', split='test')
        else:
            print(f"Skipping unknown dataset: {dataset_name}")
            continue
            
        class_names = dataset.features['label'].names
        
        # Create dataloader
        dataset.set_format(type='torch', columns=['image', 'label'])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Prepare text prompts
        text_templates = [
            "a photo of a {}",
            "a picture of a {}",
            "an image of a {}",
            "a close-up photo of a {}"
        ]
        
        text_prompts = []
        for template in text_templates:
            text_prompts.extend([template.format(c) for c in class_names])
            
        # Encode text
        with torch.no_grad():
            text_features = model.encode_text(text_prompts)
            text_features = text_features.reshape(len(text_templates), len(class_names), -1)
            text_features = text_features.mean(0)
            text_features = torch.nn.functional.normalize(text_features, dim=1)
            
        # Evaluate
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader):
                images = batch['image'].to(device)
                labels = batch['label']
                
                if transform:
                    images = torch.stack([transform(img) for img in images])
                    
                # Get image features
                image_features = model.encode_image(images)
                image_features = torch.nn.functional.normalize(image_features, dim=1)
                
                # Calculate similarity and get predictions
                similarity = image_features @ text_features.T
                preds = similarity.argmax(dim=1).cpu()
                
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())
                
        # Calculate accuracy
        accuracy = accuracy_score(all_labels, all_preds)
        
        # Store results
        results['datasets'][dataset_name] = {
            'accuracy': float(accuracy),
            'num_classes': len(class_names),
            'num_samples': len(dataset)
        }
        
        print(f"{dataset_name} Accuracy: {accuracy*100:.2f}%")
    
    return results

def plot_backbone_comparison(results_dir):
    """Create visualization comparing ResNet vs ViT performance"""
    results_files = Path(results_dir).glob('backbone_comparison_*.json')
    all_results = []
    
    for file in results_files:
        with open(file, 'r') as f:
            result = json.load(f)
            backbone = result['backbone']
            for dataset, metrics in result['datasets'].items():
                all_results.append({
                    'Backbone': backbone,
                    'Dataset': dataset,
                    'Accuracy': metrics['accuracy'] * 100
                })
                
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Dataset', y='Accuracy', hue='Backbone')
    plt.title('CLIP Zero-Shot Performance: ResNet vs ViT', pad=20)
    plt.ylabel('Accuracy (%)')
    
    # Rotate x-labels if needed
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for i in plt.gca().containers:
        plt.gca().bar_label(i, fmt='%.1f%%')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(results_dir, 'backbone_comparison.png')
    plt.savefig(plot_path)
    print(f"\n✓ Plot saved to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description='Compare CLIP backbones')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--backbone', choices=['resnet50', 'vit-b', 'vit-l'],
                      default='resnet50', help='Visual backbone to use')
    parser.add_argument('--datasets', nargs='+', 
                      default=['cifar10', 'food101', 'stl10'],
                      help='Datasets to evaluate on')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_clip_backbone(
        args.model_path,
        args.backbone,
        args.datasets,
        args.batch_size,
        args.device
    )
    
    # Save results
    results_dir = os.path.join(
        os.path.dirname(__file__), '..', 'results', 'clip'
    )
    os.makedirs(results_dir, exist_ok=True)
    
    results_path = os.path.join(
        results_dir, f'backbone_comparison_{args.backbone}.json'
    )
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_path}")
    
    # If all backbones have been evaluated, create comparison plot
    if len(list(Path(results_dir).glob('backbone_comparison_*.json'))) >= 2:
        plot_backbone_comparison(results_dir)

if __name__ == '__main__':
    main()