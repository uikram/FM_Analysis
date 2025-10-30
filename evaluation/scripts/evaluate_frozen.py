"""
Frozen Evaluation Script
Evaluates Frozen model on Flickr8k image captioning with BLEU scores

Usage:
  cd evaluation/scripts
  python evaluate_frozen.py --model_path ../../checkpoints/frozen_vision_encoder.pt
"""
import sys
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from torchvision import transforms
from tqdm import tqdm
import json
from PIL import Image

# Import from installed local package
try:
    from frozen_model import FrozenModel
except ImportError:
    print("Error: Could not import FrozenModel.")
    print("Please ensure you have run 'pip install -e ./Frozen_2106.13884v2' from the project root.")
    sys.exit(1)

# NLTK for BLEU scores
try:
    from nltk.translate.bleu_score import corpus_bleu
    from nltk.tokenize import word_tokenize
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt')
    
    # --- ADDED THIS BLOCK TO FIX THE ERROR ---
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Downloading NLTK punkt_tab resource...")
        nltk.download('punkt_tab')
    # --- END OF ADDED BLOCK ---

except ImportError:
    print("WARNING: NLTK not installed. Install with: pip install nltk")
    sys.exit(1)

# Import Hugging Face datasets
from datasets import load_dataset, Image as HFImage


class Flickr8kDataset(torch.utils.data.Dataset):
    """Dataset for Flickr8k loaded from Hugging Face"""
    def __init__(self, hf_dataset, transform, tokenizer, max_length=50):
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        
        # Load image
        image = item['image']
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = self.transform(image)
        
        # Use the first caption for loss, and all for BLEU
        caption = item['captions'][0] 
        
        # Tokenize for loss calculation
        tokens = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Prepare references for BLEU (all captions)
        references = [word_tokenize(cap.lower()) for cap in item['captions']]
        
        return {
            'image': image,
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'references': references # List of tokenized reference captions
        }


def get_frozen_dataloaders(batch_size=16):
    """Loads Flickr8k from Hugging Face and creates a test split"""
    print("Loading Flickr8k dataset...")
    try:
        # Load the dataset dictionary (which only has a 'train' split)
        dataset_dict = load_dataset("tsystems/flickr8k")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Please ensure you have an internet connection and 'datasets' installed.")
        sys.exit(1)
        
    # The tsystems/flickr8k dataset only has a 'train' split.
    # We will create a small 'test' set from this 'train' split for evaluation.
    if 'test' not in dataset_dict and 'validation' not in dataset_dict:
        print("No 'test' or 'validation' split found. Creating a 10% test split from 'train'.")
        # Create a 90/10 train/test split. We will use the 'test' part.
        split_dataset = dataset_dict['train'].train_test_split(test_size=0.1, seed=42)
        test_dataset_hf = split_dataset['test']
    else:
        # In case the dataset is updated in the future to have a proper split
        test_dataset_hf = dataset_dict.get('test', dataset_dict.get('validation'))

    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    test_dataset = Flickr8kDataset(test_dataset_hf, transform, tokenizer)
    
    # Custom collate_fn to handle list of references
    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        references = [item['references'] for item in batch] # This will be a list of lists
        return {
            'image': images,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'references': references
        }
        
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"✓ Loaded {len(test_dataset)} test samples")
    return test_loader, tokenizer


def evaluate_frozen_captioning(model_path, num_visual_tokens=2, batch_size=16, device='cuda', max_gen_length=20):
    """Evaluate Frozen on image captioning"""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading Frozen model...")
    model = FrozenModel(
        lm_model_name='gpt2', num_visual_tokens=num_visual_tokens, freeze_lm=True
    )
    
    if os.path.exists(model_path):
        # Added weights_only=True for security as suggested by the warning
        vision_state = torch.load(model_path, map_location=device, weights_only=True) 
        model.vision_encoder.load_state_dict(vision_state)
        print("✓ Loaded vision encoder weights")
    else:
        print(f"⚠ No checkpoint found at {model_path}, using untrained model")
    
    model = model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Load dataset
    test_loader, tokenizer = get_frozen_dataloaders(batch_size=batch_size)
    
    # Evaluate
    print("\nEvaluating...")
    all_references = []
    all_hypotheses = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            images = batch['image'].to(device)
            references_batch = batch['references'] # List (batch_size) of lists (5 captions) of tokens
            
            # Generate captions
            visual_embeds = model.vision_encoder(images)
            
            # Use model.language_model.generate for efficient generation
            generated_ids = model.language_model.generate(
                inputs_embeds=visual_embeds,
                max_length=max_gen_length + num_visual_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                no_repeat_ngram_size=2,
            )
            
            # Decode captions (remove visual prefix tokens)
            generated_captions = tokenizer.batch_decode(
                generated_ids[:, num_visual_tokens:], skip_special_tokens=True
            )
            
            # Tokenize hypotheses for BLEU
            for i, gen_cap in enumerate(generated_captions):
                all_hypotheses.append(word_tokenize(gen_cap.lower()))
                all_references.append(references_batch[i]) # Add the list of 5 reference captions
            
    # Calculate BLEU scores
    print("Calculating BLEU score...")
    bleu4 = corpus_bleu(all_references, all_hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    
    results = {
        'bleu4': float(bleu4),
        'parameter_stats': {
            'total_params': int(total_params),
            'trainable_params': int(trainable_params),
            'trainable_percentage': float(100 * trainable_params / total_params)
        },
        'sample_generations': [
            {
                'reference': ' '.join(all_references[i][0]), # Show first reference
                'hypothesis': ' '.join(all_hypotheses[i])
            }
            for i in range(min(5, len(all_hypotheses))) # Show 5 samples
        ]
    }
    
    # Print results
    print(f"\n{'='*70}\nFROZEN IMAGE CAPTIONING RESULTS (Flickr8k)\n{'='*70}")
    print(f"  BLEU-4: {bleu4:.4f}")
    
    print(f"\nSample Generations:")
    for i, sample in enumerate(results['sample_generations']):
        print(f"\n{i+1}. Reference: {sample['reference']}")
        print(f"   Generated: {sample['hypothesis']}")
    
    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'frozen')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_dir}/")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate Frozen on image captioning')
    parser.add_argument('--model_path', type=str,
                       default='../../checkpoints/frozen_vision_encoder.pt',
                       help='Path to vision encoder checkpoint')
    args = parser.parse_args()
    
    evaluate_frozen_captioning(args.model_path)