"""
Visual Question Answering Evaluation for Frozen Model
Tests zero-shot and few-shot VQA capabilities

Usage:
    cd evaluation/scripts
    python evaluate_frozen_vqa.py --model_path ../../checkpoints/frozen_vision_encoder.pt
"""

import os
import sys
import json
import torch
import argparse
from PIL import Image
from datasets import load_dataset
from torchvision import transforms
from transformers import AutoTokenizer
from tqdm import tqdm

# Import Frozen model
from frozen_model import FrozenModel

class VQADataset(torch.utils.data.Dataset):
    """VQAv2 dataset wrapper"""
    def __init__(self, hf_dataset, transform, tokenizer, max_length=50):
        self.dataset = hf_dataset
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Process image
        image = item['image'].convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        # Get question and answer
        question = item['question']
        if isinstance(item['answer'], list):
            # Some examples have multiple valid answers
            answer = item['answer'][0]
        else:
            answer = item['answer']
            
        return {
            'image': image,
            'question': question,
            'answer': answer
        }

def get_vqa_dataloader(split='validation', batch_size=16):
    """Load VQAv2 dataset"""
    print(f"Loading VQAv2 {split} set...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load a subset for faster evaluation
    dataset = load_dataset("HuggingFaceM4/VQAv2", split=f"{split}[:2000]")
    
    vqa_dataset = VQADataset(dataset, transform, tokenizer)
    
    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        questions = [item['question'] for item in batch]
        answers = [item['answer'] for item in batch]
        return {
            'images': images,
            'questions': questions,
            'answers': answers
        }
    
    dataloader = torch.utils.data.DataLoader(
        vqa_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print(f"✓ Loaded {len(vqa_dataset)} samples")
    return dataloader

def get_few_shot_prompt(k=4):
    """Get k-shot prompt examples for VQA"""
    # These are manually curated examples that demonstrate the VQA format
    examples = [
        {
            "question": "What color is the sky?",
            "answer": "Blue"
        },
        {
            "question": "How many people are in the image?",
            "answer": "Two people"
        },
        {
            "question": "What animal is shown?",
            "answer": "A dog"
        },
        {
            "question": "What is on the table?",
            "answer": "A cup of coffee"
        }
    ]
    
    prompt = ""
    for i, ex in enumerate(examples[:k]):
        prompt += f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n"
    
    return prompt

def evaluate_vqa(model, dataloader, device, mode='0-shot'):
    """Evaluate model on VQA task"""
    model.eval()
    
    # Get few-shot prompt if needed
    few_shot_prefix = ""
    if mode == '4-shot':
        few_shot_prefix = get_few_shot_prompt(k=4)
    
    correct = 0
    total = 0
    examples = []  # Store some examples for qualitative analysis
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating ({mode})"):
            images = batch['images'].to(device)
            questions = batch['questions']
            answers = batch['answers']
            
            # Get image embeddings
            image_embeddings = model.encode_images(images)
            
            # Process each question
            for idx, (img_embed, question, true_answer) in enumerate(zip(image_embeddings, questions, answers)):
                # Construct prompt
                prompt = few_shot_prefix + f"Question: {question}\nAnswer:"
                
                # Generate answer
                generated = model.generate_from_image(
                    img_embed.unsqueeze(0),
                    prompt,
                    max_length=50,
                    num_beams=4,
                    temperature=0.7,
                    top_p=0.9
                )
                
                pred_answer = generated.strip()
                
                # Simple exact match accuracy
                if pred_answer.lower() == true_answer.lower():
                    correct += 1
                
                # Store example (for first batch only)
                if len(examples) < 5:
                    examples.append({
                        'question': question,
                        'true_answer': true_answer,
                        'pred_answer': pred_answer,
                        'correct': pred_answer.lower() == true_answer.lower()
                    })
                
                total += 1
    
    accuracy = correct / total
    return accuracy, examples

def evaluate_frozen_vqa(model_path, num_visual_tokens=2, batch_size=16, device='cuda'):
    """Evaluate Frozen model on VQA in both 0-shot and 4-shot settings"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading Frozen model...")
    model = FrozenModel(
        lm_model_name='gpt2',
        num_visual_tokens=num_visual_tokens,
        freeze_lm=True
    )
    
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        sys.exit(1)
        
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    
    # Load validation data
    val_loader = get_vqa_dataloader(batch_size=batch_size)
    
    # Evaluate in both modes
    results = {}
    for mode in ['0-shot', '4-shot']:
        accuracy, examples = evaluate_vqa(model, val_loader, device, mode)
        
        results[mode] = {
            'accuracy': float(accuracy),
            'examples': examples
        }
        
        print(f"\n{mode.upper()} RESULTS:")
        print(f"  Accuracy: {accuracy*100:.2f}%")
        
        print("\nExample Predictions:")
        for ex in examples[:3]:  # Show first 3 examples
            print(f"\nQ: {ex['question']}")
            print(f"A (true): {ex['true_answer']}")
            print(f"A (pred): {ex['pred_answer']}")
            print(f"Correct: {ex['correct']}")
    
    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'frozen')
    os.makedirs(results_dir, exist_ok=True)
    
    results_path = os.path.join(results_dir, 'vqa_evaluation.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\n✓ Results saved to {results_path}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Frozen on VQA")
    parser.add_argument('--model_path', required=True, help='Path to trained Frozen model')
    parser.add_argument('--num_visual_tokens', type=int, default=2,
                      help='Number of visual tokens to use')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', default='cuda', help='Device to run on (cuda/cpu)')
    args = parser.parse_args()
    
    evaluate_frozen_vqa(
        args.model_path,
        args.num_visual_tokens,
        args.batch_size,
        args.device
    )