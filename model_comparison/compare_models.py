"""
Comprehensive Model Comparison Script
Compares CLIP, LoRA, and Frozen models on various metrics
"""

import sys
import os
sys.path.append('../CLIP_radford21a/src')
sys.path.append('../LoRA_2106.09685v1/src')
sys.path.append('../Frozen_2106.13884v2/src')

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from time import time
import json

# Import models
from clip_model import CLIP
from lora_model import LoRALinear, LoRAForLanguageModel
from frozen_model import FrozenModel

def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def measure_inference_time(model, input_data, num_runs=100):
    """Measure average inference time"""
    model.eval()
    times = []
    
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(**input_data)
        
        # Measure
        for _ in range(num_runs):
            start = time()
            _ = model(**input_data)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times.append(time() - start)
    
    return np.mean(times), np.std(times)

def compare_parameter_efficiency():
    """Compare parameter efficiency across models"""
    print("\n" + "="*60)
    print("PARAMETER EFFICIENCY COMPARISON")
    print("="*60)
    
    results = {}
    
    # CLIP
    print("\n[1/3] Analyzing CLIP...")
    clip_model = CLIP(embed_dim=512)
    clip_total, clip_trainable = count_parameters(clip_model)
    results['CLIP'] = {
        'total': clip_total,
        'trainable': clip_trainable,
        'percentage': 100 * clip_trainable / clip_total
    }
    print(f"  Total parameters: {clip_total:,}")
    print(f"  Trainable parameters: {clip_trainable:,}")
    
    # LoRA (simulated with GPT-2)
    print("\n[2/3] Analyzing LoRA...")
    from transformers import GPT2LMHeadModel
    base_model = GPT2LMHeadModel.from_pretrained('gpt2')
    lora_model = LoRAForLanguageModel(base_model, rank=4)
    lora_total, lora_trainable = count_parameters(lora_model)
    results['LoRA'] = {
        'total': lora_total,
        'trainable': lora_trainable,
        'percentage': 100 * lora_trainable / lora_total
    }
    print(f"  Total parameters: {lora_total:,}")
    print(f"  Trainable parameters: {lora_trainable:,}")
    print(f"  Reduction: {lora_total / lora_trainable:.1f}x")
    
    # Frozen
    print("\n[3/3] Analyzing Frozen...")
    frozen_model = FrozenModel(lm_model_name='gpt2', num_visual_tokens=2)
    frozen_total, frozen_trainable = count_parameters(frozen_model)
    results['Frozen'] = {
        'total': frozen_total,
        'trainable': frozen_trainable,
        'percentage': 100 * frozen_trainable / frozen_total
    }
    print(f"  Total parameters: {frozen_total:,}")
    print(f"  Trainable parameters: {frozen_trainable:,}")
    
    return results

def visualize_comparison(param_results):
    """Create comparison visualizations"""
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Comparison: CLIP vs LoRA vs Frozen', fontsize=16, fontweight='bold')
    
    models = list(param_results.keys())
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    # 1. Total vs Trainable Parameters
    ax1 = axes[0, 0]
    x = np.arange(len(models))
    width = 0.35
    
    total_params = [param_results[m]['total'] / 1e6 for m in models]
    trainable_params = [param_results[m]['trainable'] / 1e6 for m in models]
    
    ax1.bar(x - width/2, total_params, width, label='Total', color=colors, alpha=0.8)
    ax1.bar(x + width/2, trainable_params, width, label='Trainable', color=colors, alpha=0.5)
    ax1.set_ylabel('Parameters (Millions)', fontsize=12)
    ax1.set_title('Total vs Trainable Parameters', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Percentage of Trainable Parameters
    ax2 = axes[0, 1]
    percentages = [param_results[m]['percentage'] for m in models]
    bars = ax2.barh(models, percentages, color=colors)
    ax2.set_xlabel('Trainable Parameters (%)', fontsize=12)
    ax2.set_title('Parameter Efficiency', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, pct) in enumerate(zip(bars, percentages)):
        ax2.text(pct + 1, i, f'{pct:.2f}%', va='center', fontsize=10)
    
    # 3. Architecture Comparison Table
    ax3 = axes[1, 0]
    ax3.axis('off')
    
    table_data = []
    for model in models:
        table_data.append([
            model,
            f"{param_results[model]['total']:,}",
            f"{param_results[model]['trainable']:,}",
            f"{param_results[model]['percentage']:.2f}%"
        ])
    
    table = ax3.table(
        cellText=table_data,
        colLabels=['Model', 'Total Params', 'Trainable Params', '% Trainable'],
        cellLoc='center',
        loc='center',
        colWidths=[0.2, 0.25, 0.3, 0.25]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 4. Key Features Comparison
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    features_text = """
    Key Characteristics:
    
    CLIP:
    • Contrastive vision-language learning
    • Zero-shot transfer capabilities
    • All parameters trainable
    • Use case: Image-text matching
    
    LoRA:
    • Parameter-efficient fine-tuning
    • 10,000x parameter reduction
    • No inference latency
    • Use case: Task adaptation
    
    Frozen:
    • Frozen language model backbone
    • Vision encoder training only
    • Few-shot multimodal learning
    • Use case: Visual QA
    """
    
    ax4.text(0.1, 0.5, features_text, fontsize=10, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('figures/model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved visualization to figures/model_comparison.png")
    
    return fig

def create_summary_report(param_results):
    """Create a summary report"""
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)
    
    report = {
        'comparison_date': str(pd.Timestamp.now()),
        'models': param_results,
        'key_findings': {
            'most_efficient': min(param_results.keys(), 
                                 key=lambda x: param_results[x]['percentage']),
            'largest_model': max(param_results.keys(), 
                                key=lambda x: param_results[x]['total']),
        }
    }
    
    # Save report
    with open('results/comparison_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("\n✓ Saved report to results/comparison_report.json")
    
    # Print summary
    print("\nKEY FINDINGS:")
    print(f"  Most parameter-efficient: {report['key_findings']['most_efficient']}")
    print(f"  Largest model: {report['key_findings']['largest_model']}")
    
    return report

def main():
    """Run complete comparison"""
    print("\n" + "="*60)
    print("MODEL COMPARISON ANALYSIS")
    print("CLIP vs LoRA vs Frozen")
    print("="*60)
    
    # Create output directories
    os.makedirs('figures', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Run comparisons
    param_results = compare_parameter_efficiency()
    
    # Generate visualizations
    fig = visualize_comparison(param_results)
    
    # Create summary report
    report = create_summary_report(param_results)
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE!")
    print("="*60)
    print("\nOutputs:")
    print("  • figures/model_comparison.png")
    print("  • results/comparison_report.json")
    
    return param_results, report

if __name__ == "__main__":
    results, report = main()
