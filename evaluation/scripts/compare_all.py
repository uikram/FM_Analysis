"""Compare all three models and generate visualization"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def load_results():
    """Load all evaluation results"""
    clip_results = json.load(open('../results/clip/evaluation_results.json'))
    lora_results = json.load(open('../results/lora/evaluation_results.json'))
    frozen_results = json.load(open('../results/frozen/evaluation_results.json'))
    return clip_results, lora_results, frozen_results

def create_comparison_plots():
    """Generate comparison visualizations"""
    clip_res, lora_res, frozen_res = load_results()
    
    # Create comparison table
    comparison_data = {
        'Model': ['CLIP', 'LoRA', 'Frozen'],
        'Primary Metric': [
            f"{clip_res['accuracy']*100:.1f}% (Top-1 Acc)",
            f"{lora_res['accuracy']*100:.1f}% (Accuracy)", 
            f"{frozen_res['bleu4']:.3f} (BLEU-4)"
        ],
        'Total Params': ['~100M', '~125M', '~7B'],
        'Trainable Params': ['100M (100%)', '0.3M (0.24%)', '25M (0.36%)'],
        'Efficiency': ['1x', '10,000x', '280x']
    }
    
    df = pd.DataFrame(comparison_data)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Parameter efficiency
    models = ['CLIP', 'LoRA', 'Frozen']
    trainable_pcts = [100, 0.24, 0.36]
    axes[0,0].bar(models, trainable_pcts)
    axes[0,0].set_title('Parameter Efficiency (% Trainable)')
    axes[0,0].set_ylabel('Trainable %')
    
    # Plot 2: Performance comparison
    # ... (add more plots)
    
    plt.tight_layout()
    plt.savefig('../visualizations/model_comparison.png', dpi=300)
    print("✓ Saved comparison plots")

if __name__ == "__main__":
    create_comparison_plots()
