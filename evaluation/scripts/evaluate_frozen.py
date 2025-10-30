"""
Frozen Evaluation Script with Fixed Imports
Evaluates Frozen model on image captioning with BLEU scores

Usage:
  cd evaluation/scripts
  CUDA_VISIBLE_DEVICES=2 python evaluate_frozen.py --model_path ../../Frozen_2106.13884v2/checkpoints/frozen_vision_encoder.pt
"""

import sys
import os

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
frozen_src = os.path.join(project_root, 'Frozen_2106.13884v2', 'src')
eval_datasets = os.path.join(current_dir, '..', 'datasets')

sys.path.insert(0, frozen_src)
sys.path.insert(0, eval_datasets)

import torch
from frozen_model import FrozenModel
from frozen_datasets import get_frozen_dataloaders
from transformers import AutoTokenizer
from tqdm import tqdm
import json

# NLTK for BLEU scores
try:
    from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
    from nltk.tokenize import word_tokenize
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt')
except ImportError:
    print("WARNING: NLTK not installed. Install with: pip install nltk")
    print("BLEU scores will not be available.")

def evaluate_frozen_captioning(model_path, num_visual_tokens=2, batch_size=16, device='cuda', max_gen_length=20):
    """Evaluate Frozen on image captioning"""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    print("\nLoading Frozen model...")
    model = FrozenModel(
        lm_model_name='gpt2',
        num_visual_tokens=num_visual_tokens,
        freeze_lm=True
    )
    
    # Load vision encoder weights if available
    if os.path.exists(model_path):
        vision_state = torch.load(model_path, map_location=device)
        model.vision_encoder.load_state_dict(vision_state)
        print("✓ Loaded vision encoder weights")
    else:
        print("⚠ No checkpoint found, using untrained model")
    
    model = model.to(device)
    model.eval()
    
    # Parameter stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParameter Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable (vision encoder): {trainable_params:,}")
    print(f"  Frozen (LM): {total_params - trainable_params:,}")
    print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print("\nLoading dataset...")
    _, test_loader = get_frozen_dataloaders(batch_size=batch_size, use_simple=True)
    print(f"✓ Loaded {len(test_loader.dataset)} test samples")
    
    # Evaluate
    print("\nEvaluating...")
    all_references = []
    all_hypotheses = []
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            images = batch['image'].to(device)
            captions = batch['caption']
            
            # Tokenize captions for loss calculation
            caption_tokens = tokenizer(
                captions,
                padding='max_length',
                truncation=True,
                max_length=50,
                return_tensors='pt'
            ).to(device)
            
            # Forward pass for loss
            outputs = model(
                images=images,
                input_ids=caption_tokens['input_ids'],
                attention_mask=caption_tokens['attention_mask'],
                labels=caption_tokens['input_ids']
            )
            
            total_loss += outputs.loss.item()
            num_batches += 1
            
            # Generate captions for BLEU evaluation
            visual_embeds = model.vision_encoder(images)
            batch_size_actual = visual_embeds.size(0)
            
            # Simple greedy generation
            for i in range(batch_size_actual):
                gen_tokens = []
                current_embeds = visual_embeds[i:i+1]
                
                for _ in range(max_gen_length):
                    outputs = model.language_model(inputs_embeds=current_embeds)
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token_id = torch.argmax(next_token_logits, dim=-1)
                    
                    gen_tokens.append(next_token_id.item())
                    
                    # Stop if EOS token
                    if next_token_id.item() == tokenizer.eos_token_id:
                        break
                    
                    # Append new token embedding
                    next_token_embed = model.text_embeddings(next_token_id.unsqueeze(0))
                    current_embeds = torch.cat([current_embeds, next_token_embed], dim=1)
                
                # Decode generated caption
                generated_caption = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                
                # Tokenize for BLEU
                if 'word_tokenize' in globals():
                    ref_tokens = word_tokenize(captions[i].lower())
                    hyp_tokens = word_tokenize(generated_caption.lower())
                else:
                    ref_tokens = captions[i].lower().split()
                    hyp_tokens = generated_caption.lower().split()
                
                all_references.append([ref_tokens])
                all_hypotheses.append(hyp_tokens)
    
    # Calculate metrics
    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    # Calculate BLEU scores if NLTK is available
    if 'corpus_bleu' in globals():
        bleu1 = corpus_bleu(all_references, all_hypotheses, weights=(1, 0, 0, 0))
        bleu2 = corpus_bleu(all_references, all_hypotheses, weights=(0.5, 0.5, 0, 0))
        bleu3 = corpus_bleu(all_references, all_hypotheses, weights=(0.33, 0.33, 0.33, 0))
        bleu4 = corpus_bleu(all_references, all_hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    else:
        bleu1 = bleu2 = bleu3 = bleu4 = 0.0
        print("WARNING: BLEU scores not calculated (NLTK not available)")
    
    results = {
        'perplexity': float(perplexity),
        'bleu1': float(bleu1),
        'bleu2': float(bleu2),
        'bleu3': float(bleu3),
        'bleu4': float(bleu4),
        'parameter_stats': {
            'total_params': int(total_params),
            'trainable_params': int(trainable_params),
            'frozen_params': int(total_params - trainable_params),
            'trainable_percentage': float(100 * trainable_params / total_params)
        },
        'sample_generations': [
            {
                'reference': ' '.join(all_references[i][0]),
                'hypothesis': ' '.join(all_hypotheses[i])
            }
            for i in range(min(10, len(all_references)))
        ]
    }
    
    # Print results
    print(f"\n{'='*70}")
    print("FROZEN IMAGE CAPTIONING RESULTS")
    print(f"{'='*70}")
    
    print(f"\nOverall Metrics:")
    print(f"  Perplexity: {perplexity:.2f}")
    if bleu1 > 0:
        print(f"  BLEU-1: {bleu1:.4f}")
        print(f"  BLEU-2: {bleu2:.4f}")
        print(f"  BLEU-3: {bleu3:.4f}")
        print(f"  BLEU-4: {bleu4:.4f}")
    
    print(f"\nParameter Efficiency:")
    print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"  Frozen LM: {total_params - trainable_params:,}")
    
    print(f"\nSample Generations:")
    for i, sample in enumerate(results['sample_generations'][:5]):
        print(f"\n{i+1}. Reference: {sample['reference']}")
        print(f"   Generated: {sample['hypothesis']}")
    
    # Save results
    results_dir = os.path.join(current_dir, '..', 'results', 'frozen')
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_dir}/")
    print(f"  - evaluation_results.json")
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate Frozen on image captioning')
    parser.add_argument('--model_path', type=str,
                       default='../../Frozen_2106.13884v2/checkpoints/frozen_vision_encoder.pt',
                       help='Path to vision encoder checkpoint')
    parser.add_argument('--num_visual_tokens', type=int, default=2,
                       help='Number of visual tokens')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--max_gen_length', type=int, default=20,
                       help='Maximum caption generation length')
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    evaluate_frozen_captioning(
        args.model_path, args.num_visual_tokens, args.batch_size, 
        args.device, args.max_gen_length
    )
