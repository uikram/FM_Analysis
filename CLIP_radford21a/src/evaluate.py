"""
Evaluation script for CLIP model
Tests zero-shot classification and image retrieval
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import json

from clip_model import CLIP
from transformers import AutoTokenizer

class CLIPEvaluator:
    """Evaluator for CLIP model"""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
    
    def zero_shot_classify(self, images, class_names, templates=None):
        """
        Perform zero-shot classification
        
        Args:
            images: Batch of images
            class_names: List of class names
            templates: Optional templates like "a photo of a {}"
        """
        if templates is None:
            templates = ["a photo of a {}"]
        
        # Create text prompts for each class
        texts = []
        for class_name in class_names:
            for template in templates:
                texts.append(template.format(class_name))
        
        # Tokenize texts
        text_tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            # Encode images
            image_features = self.model.encode_image(images)
            
            # Encode texts
            text_features = self.model.encode_text(
                text_tokens['input_ids'],
                text_tokens['attention_mask']
            )
            
            # Compute similarity
            similarity = image_features @ text_features.T
            
            # Average over templates
            if len(templates) > 1:
                similarity = similarity.view(
                    len(images),
                    len(class_names),
                    len(templates)
                ).mean(dim=2)
            
            # Get predictions
            probs = F.softmax(similarity, dim=1)
            predictions = similarity.argmax(dim=1)
        
        return predictions, probs
    
    def evaluate_dataset(self, dataloader, class_names):
        """
        Evaluate on a dataset
        
        Args:
            dataloader: DataLoader with images and labels
            class_names: List of class names
        """
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch in tqdm(dataloader, desc='Evaluating'):
            images = batch['images'].to(self.device)
            labels = batch['labels']
            
            predictions, probs = self.zero_shot_classify(images, class_names)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        top5_acc = top_k_accuracy_score(
            all_labels,
            np.array(all_probs),
            k=5,
            labels=range(len(class_names))
        )
        
        results = {
            'accuracy': float(accuracy),
            'top5_accuracy': float(top5_acc),
            'num_samples': len(all_labels),
            'num_classes': len(class_names)
        }
        
        return results
    
    def image_text_retrieval(self, images, texts, k=5):
        """
        Perform image-to-text and text-to-image retrieval
        
        Args:
            images: Batch of images
            texts: List of text captions
            k: Number of top retrievals
        """
        # Tokenize texts
        text_tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            # Encode
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(
                text_tokens['input_ids'],
                text_tokens['attention_mask']
            )
            
            # Compute similarity
            similarity = image_features @ text_features.T
            
            # Image-to-text retrieval
            i2t_indices = similarity.topk(k, dim=1).indices
            
            # Text-to-image retrieval
            t2i_indices = similarity.T.topk(k, dim=1).indices
        
        return i2t_indices, t2i_indices

def run_evaluation(checkpoint_path, eval_config):
    """
    Run complete evaluation
    
    Args:
        checkpoint_path: Path to model checkpoint
        eval_config: Configuration dict with dataset info
    """
    print("Loading model...")
    model = CLIP(embed_dim=512)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    evaluator = CLIPEvaluator(model, tokenizer, device)
    
    results = {}
    
    print("\nEvaluation Results:")
    print("="*60)
    
    # Note: This is a template - actual evaluation requires dataset loading
    print("\n[Note] This is an evaluation template.")
    print("Replace with actual dataset loading and evaluation.")
    
    return results

if __name__ == "__main__":
    print("CLIP Model Evaluation")
    print("="*60)
    print("\nUsage: python evaluate.py --checkpoint path/to/checkpoint.pt")
