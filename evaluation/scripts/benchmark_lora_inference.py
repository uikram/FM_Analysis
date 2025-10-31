"""
Benchmark LoRA Inference Speed
Compares standard vs merged weight inference latency

Usage:
    python benchmark_lora_inference.py --model_path ../../LoRA_2106.09685v1/checkpoints/lora_best.pt
"""

import os
import sys
import time
import torch
import numpy as np
import argparse
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import local packages
from lora_model import apply_lora_to_model
from weight_merging import create_merged_model
from evaluate_lora import get_lora_dataloaders

def time_inference(model, test_loader, num_batches=100, device='cuda'):
    """Time model inference over multiple batches"""
    model.eval()
    latencies = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= num_batches:
                break
                
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Time the forward pass
            start_time = time.perf_counter()
            _ = model(**batch)
            end_time = time.perf_counter()
            
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
    
    return {
        'mean_latency': np.mean(latencies),
        'std_latency': np.std(latencies),
        'p90_latency': np.percentile(latencies, 90),
        'p95_latency': np.percentile(latencies, 95),
    }

def benchmark_inference(model_path, batch_size=32, device='cuda'):
    """Compare inference speed of standard vs merged LoRA"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load base model and apply LoRA
    print("\nLoading models...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=2
    )
    lora_model = apply_lora_to_model(base_model)
    
    # Load trained weights
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        sys.exit(1)
        
    lora_model.load_state_dict(torch.load(model_path))
    lora_model = lora_model.to(device)
    
    # Create merged version
    print("Creating merged model...")
    merged_model = create_merged_model(lora_model)
    merged_model = merged_model.to(device)
    
    # Load test data
    test_loader = get_lora_dataloaders(batch_size=batch_size)
    
    # Warmup
    print("\nWarming up models...")
    _ = time_inference(lora_model, test_loader, num_batches=10, device=device)
    _ = time_inference(merged_model, test_loader, num_batches=10, device=device)
    
    # Benchmark
    print("\nBenchmarking standard LoRA...")
    lora_stats = time_inference(lora_model, test_loader, device=device)
    
    print("Benchmarking merged LoRA...")
    merged_stats = time_inference(merged_model, test_loader, device=device)
    
    # Report results
    print(f"\n{'='*70}\nLoRA INFERENCE BENCHMARK RESULTS\n{'='*70}")
    print("\nStandard LoRA (BA computation):")
    print(f"  Mean latency: {lora_stats['mean_latency']:.2f}ms")
    print(f"  P90 latency: {lora_stats['p90_latency']:.2f}ms")
    print(f"  P95 latency: {lora_stats['p95_latency']:.2f}ms")
    
    print("\nMerged LoRA (single matrix):")
    print(f"  Mean latency: {merged_stats['mean_latency']:.2f}ms")
    print(f"  P90 latency: {merged_stats['p90_latency']:.2f}ms")
    print(f"  P95 latency: {merged_stats['p95_latency']:.2f}ms")
    
    speedup = (lora_stats['mean_latency'] / merged_stats['mean_latency'] - 1) * 100
    print(f"\nSpeedup from merging: {speedup:.1f}%")
    
    # Save results
    results = {
        'standard_lora': lora_stats,
        'merged_lora': merged_stats,
        'speedup_percentage': speedup,
        'batch_size': batch_size
    }
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'lora')
    os.makedirs(results_dir, exist_ok=True)
    
    benchmark_path = os.path.join(results_dir, 'inference_benchmark.json')
    import json
    with open(benchmark_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\n✓ Benchmark results saved to {benchmark_path}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark LoRA inference")
    parser.add_argument('--model_path', required=True, help='Path to trained LoRA model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', default='cuda', help='Device to run on (cuda/cpu)')
    args = parser.parse_args()
    
    benchmark_inference(args.model_path, args.batch_size, args.device)