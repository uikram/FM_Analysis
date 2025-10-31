"""
Enhanced CLIP Evaluation Script
Tests CLIP zero-shot transfer across multiple image classification datasets
"""

import os
import sys
import json
import torch
import argparse
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# Import CLIP model
from clip_model import CLIP

class ImageClassificationDataset(torch.utils.data.Dataset):
    """Generic dataset for HuggingFace image classification datasets"""
    def __init__(self, hf_dataset, transform, class_names):
        self.dataset = hf_dataset
        self.transform = transform
        self.class_names = class_names
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert('RGB')
        label = item['label']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transform():
    """Standard CLIP preprocessing"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def evaluate_on_dataset(model, dataset_name, device):
    """Evaluate CLIP zero-shot classification on a specific dataset"""
    print(f"\nEvaluating on {dataset_name}...")
    
    # Load dataset
    if dataset_name == "food101":
        dataset = load_dataset("food101", split="validation[:2000]")  # Subsample for speed
        class_names = dataset.features['label'].names
        
    elif dataset_name == "stl10":
        dataset = load_dataset("stl10", split="test")
        class_names = dataset.features['label'].names
        
    elif dataset_name == "cifar10":
        dataset = load_dataset("cifar10", split="test")
        class_names = dataset.features['label'].names
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create dataloader
    test_dataset = ImageClassificationDataset(
        dataset, transform=get_transform(), class_names=class_names
    )
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=4
    )
    
    print(f"✓ Loaded {len(test_dataset)} test images")
    print(f"Classes: {class_names}")
    
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
        
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    text_tokens = tokenizer(
        text_prompts, 
        padding=True, 
        truncation=True, 
        max_length=77, 
        return_tensors='pt'
    ).to(device)
    
    # Encode text descriptions
    print("Encoding text descriptions...")
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features.reshape(
            len(text_templates), len(class_names), -1
        ).mean(0)
        text_features = torch.nn.functional.normalize(text_features, dim=1)
    
    # Evaluate
    print("Running inference...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            
            # Get image features
            image_features = model.encode_image(images)
            image_features = torch.nn.functional.normalize(image_features, dim=1)
            
            # Calculate similarity
            similarity = image_features @ text_features.T
            
            # Get predictions
            preds = similarity.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, 
        target_names=class_names, 
        output_dict=True
    )
    
    results = {
        'dataset': dataset_name,
        'accuracy': float(accuracy),
        'classification_report': report,
        'num_classes': len(class_names),
        'num_samples': len(test_dataset)
    }
    
    print(f"\n{dataset_name.upper()} RESULTS:")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  Classes: {len(class_names)}")
    print(f"  Samples: {len(test_dataset)}")
    
    return results

def evaluate_clip_transfer(model_path, device='cuda'):
    """Evaluate CLIP zero-shot transfer on multiple datasets"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading CLIP model...")
    model = CLIP(embed_dim=512)
    
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        sys.exit(1)
        
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    # Evaluate on each dataset
    datasets = ['cifar10', 'food101', 'stl10']
    all_results = {}
    
    for dataset in datasets:
        results = evaluate_on_dataset(model, dataset, device)
        all_results[dataset] = results
    
    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'clip')
    os.makedirs(results_dir, exist_ok=True)
    
    results_path = os.path.join(results_dir, 'transfer_evaluation.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
        
    print(f"\n✓ Results saved to {results_path}")
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CLIP zero-shot transfer")
    parser.add_argument('--model_path', required=True, help='Path to trained CLIP model')
    parser.add_argument('--device', default='cuda', help='Device to run on (cuda/cpu)')
    args = parser.parse_args()
    
    evaluate_clip_transfer(args.model_path, args.device)