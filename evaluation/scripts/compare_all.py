"""
Complete Model Comparison and Visualization Script
Loads evaluation results from all three models and generates comprehensive comparisons

Usage:
  cd evaluation/scripts
  python compare_all.py
"""

import json
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

current_dir = os.path.dirname(os.path.abspath(__file__))
results_base = os.path.join(current_dir, '..', 'results')
viz_dir = os.path.join(current_dir, '..', 'visualizations')
os.makedirs(viz_dir, exist_ok=True)

def load_all_results():
    """Load evaluation results from all three models"""
    print("Loading evaluation results...")
    
    results = {}
    
    # Load CLIP results
    clip_path = os.path.join(results_base, 'clip', 'evaluation_results.json')
    if os.path.exists(clip_path):
        with open(clip_path, 'r') as f:
            results['clip'] = json.load(f)
        print("✓ Loaded CLIP results")
    else:
        print("⚠ CLIP results not found. Run evaluate_clip.py first.")
        results['clip'] = None
    
    # Load LoRA results
    lora_path = os.path.join(results_base, 'lora', 'evaluation_results.json')
    if os.path.exists(lora_path):
        with open(lora_path, 'r') as f:
            results['lora'] = json.load(f)
        print("✓ Loaded LoRA results")
    else:
        print("⚠ LoRA results not found. Run evaluate_lora.py first.")
        results['lora'] = None
    
    # Load Frozen results
    frozen_path = os.path.join(results_base, 'frozen', 'evaluation_results.json')
    if os.path.exists(frozen_path):
        with open(frozen_path, 'r') as f:
            results['frozen'] = json.load(f)
        print("✓ Loaded Frozen results")
    else:
        print("⚠ Frozen results not found. Run evaluate_frozen.py first.")
        results['frozen'] = None
    
    return results

def create_comparison_table(results):
    """Create comprehensive comparison table"""
    print("\nGenerating comparison table...")
    
    comparison_data = []
    
    # CLIP data
    if results['clip']:
        comparison_data.append({
            'Model': 'CLIP',
            'Task': 'Zero-shot Classification',
            'Dataset': 'CIFAR-10',
            'Primary Metric': f"{results['clip']['accuracy']*100:.2f}%",
            'Secondary Metric': f"Top-5: {results['clip']['top5_accuracy']*100:.2f}%",
            'Total Params': '~100M',
            'Trainable Params': '100M',
            'Trainable %': '100%',
            'Efficiency Factor': '1x'
        })
    
    # LoRA data
    if results['lora']:
        comparison_data.append({
            'Model': 'LoRA',
            'Task': 'Sentiment Classification',
            'Dataset': 'IMDB',
            'Primary Metric': f"{results['lora']['accuracy']*100:.2f}%",
            'Secondary Metric': f"F1: {results['lora']['f1_score']:.4f}",
            'Total Params': f"{results['lora']['parameter_stats']['total_params']:,}",
            'Trainable Params': f"{results['lora']['parameter_stats']['trainable_params']:,}",
            'Trainable %': f"{results['lora']['parameter_stats']['trainable_percentage']:.2f}%",
            'Efficiency Factor': f"{results['lora']['parameter_stats']['reduction_factor']:.0f}x"
        })
    
    # Frozen data
    if results['frozen']:
        comparison_data.append({
            'Model': 'Frozen',
            'Task': 'Image Captioning',
            'Dataset': 'Synthetic/Flickr8k',
            'Primary Metric': f"BLEU-4: {results['frozen']['bleu4']:.4f}",
            'Secondary Metric': f"Perplexity: {results['frozen']['perplexity']:.2f}",
            'Total Params': f"{results['frozen']['parameter_stats']['total_params']:,}",
            'Trainable Params': f"{results['frozen']['parameter_stats']['trainable_params']:,}",
            'Trainable %': f"{results['frozen']['parameter_stats']['trainable_percentage']:.2f}%",
            'Efficiency Factor': f"{100/results['frozen']['parameter_stats']['trainable_percentage']:.0f}x"
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    csv_path = os.path.join(results_base, 'model_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved comparison table to {csv_path}")
    
    return df

def plot_parameter_efficiency(results):
    """Plot parameter efficiency comparison"""
    print("Creating parameter efficiency plot...")
    
    models = []
    trainable_pcts = []
    total_params = []
    
    if results['clip']:
        models.append('CLIP')
        trainable_pcts.append(100.0)
        total_params.append(100)  # Million
    
    if results['lora']:
        models.append('LoRA')
        trainable_pcts.append(results['lora']['parameter_stats']['trainable_percentage'])
        total_params.append(results['lora']['parameter_stats']['total_params'] / 1e6)
    
    if results['frozen']:
        models.append('Frozen')
        trainable_pcts.append(results['frozen']['parameter_stats']['trainable_percentage'])
        total_params.append(results['frozen']['parameter_stats']['total_params'] / 1e6)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Trainable percentage
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars1 = ax1.bar(models, trainable_pcts, color=colors[:len(models)])
    ax1.set_ylabel('Trainable Parameters (%)', fontsize=12)
    ax1.set_title('Parameter Efficiency Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 105)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom')
    
    # Plot 2: Total parameters (log scale)
    ax2.bar(models, total_params, color=colors[:len(models)])
    ax2.set_ylabel('Total Parameters (Millions, log scale)', fontsize=12)
    ax2.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    
    # Add value labels
    for i, (model, params) in enumerate(zip(models, total_params)):
        ax2.text(i, params, f'{params:.1f}M', ha='center', va='bottom')
    
    plt.tight_layout()
    save_path = os.path.join(viz_dir, 'parameter_efficiency.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {save_path}")
    plt.close()

def plot_confusion_matrices(results):
    """Plot confusion matrices for CLIP and LoRA"""
    print("Creating confusion matrix plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # CLIP confusion matrix
    if results['clip']:
        clip_cm = np.array(results['clip']['confusion_matrix'])
        class_names = list(results['clip']['per_class_accuracy'].keys())
        
        sns.heatmap(clip_cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[0])
        axes[0].set_title('CLIP Confusion Matrix (CIFAR-10)', fontweight='bold')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
    
    # LoRA confusion matrix
    if results['lora']:
        lora_cm = np.array(results['lora']['confusion_matrix'])
        labels = ['Negative', 'Positive']
        
        sns.heatmap(lora_cm, annot=True, fmt='d', cmap='Greens',
                   xticklabels=labels, yticklabels=labels,
                   ax=axes[1])
        axes[1].set_title('LoRA Confusion Matrix (IMDB)', fontweight='bold')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    save_path = os.path.join(viz_dir, 'confusion_matrices.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {save_path}")
    plt.close()

def plot_performance_summary(results):
    """Create performance summary visualization"""
    print("Creating performance summary plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Accuracy comparison (where applicable)
    models = []
    accuracies = []
    
    if results['clip']:
        models.append('CLIP\n(Top-1)')
        accuracies.append(results['clip']['accuracy'] * 100)
    
    if results['lora']:
        models.append('LoRA\n(Binary)')
        accuracies.append(results['lora']['accuracy'] * 100)
    
    if models:
        axes[0, 0].bar(models, accuracies, color=['#3498db', '#e74c3c'])
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].set_title('Classification Accuracy', fontweight='bold')
        axes[0, 0].set_ylim(0, 100)
        
        for i, (model, acc) in enumerate(zip(models, accuracies)):
            axes[0, 0].text(i, acc + 2, f'{acc:.1f}%', ha='center')
    
    # 2. Efficiency factor
    if results['lora'] and results['frozen']:
        models_eff = ['LoRA', 'Frozen']
        efficiency = [
            results['lora']['parameter_stats']['reduction_factor'],
            100 / results['frozen']['parameter_stats']['trainable_percentage']
        ]
        
        axes[0, 1].bar(models_eff, efficiency, color=['#e74c3c', '#2ecc71'])
        axes[0, 1].set_ylabel('Efficiency Factor (x)')
        axes[0, 1].set_title('Parameter Efficiency (Higher = Better)', fontweight='bold')
        axes[0, 1].set_yscale('log')
        
        for i, (model, eff) in enumerate(zip(models_eff, efficiency)):
            axes[0, 1].text(i, eff, f'{eff:.0f}x', ha='center', va='bottom')
    
    # 3. Per-class performance for CLIP
    if results['clip']:
        per_class = results['clip']['per_class_accuracy']
        classes = list(per_class.keys())
        accs = [per_class[c] * 100 for c in classes]
        
        axes[1, 0].barh(classes, accs, color='#3498db')
        axes[1, 0].set_xlabel('Accuracy (%)')
        axes[1, 0].set_title('CLIP Per-Class Accuracy', fontweight='bold')
        axes[1, 0].set_xlim(0, 100)
    
    # 4. LoRA metrics breakdown
    if results['lora']:
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [
            results['lora']['accuracy'] * 100,
            results['lora']['precision'] * 100,
            results['lora']['recall'] * 100,
            results['lora']['f1_score'] * 100
        ]
        
        axes[1, 1].bar(metrics, values, color='#e74c3c')
        axes[1, 1].set_ylabel('Score (%)')
        axes[1, 1].set_title('LoRA Performance Metrics', fontweight='bold')
        axes[1, 1].set_ylim(0, 100)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        for i, (metric, val) in enumerate(zip(metrics, values)):
            axes[1, 1].text(i, val + 2, f'{val:.1f}%', ha='center')
    
    plt.tight_layout()
    save_path = os.path.join(viz_dir, 'performance_summary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {save_path}")
    plt.close()

def generate_report(results, df):
    """Generate text report"""
    print("\nGenerating text report...")
    
    report = []
    report.append("="*70)
    report.append("FOUNDATION MODELS EVALUATION REPORT")
    report.append("="*70)
    report.append("")
    
    # Summary table
    report.append("COMPARISON SUMMARY")
    report.append("-"*70)
    report.append(df.to_string(index=False))
    report.append("")
    
    # Detailed results
    report.append("DETAILED RESULTS")
    report.append("-"*70)
    
    if results['clip']:
        report.append("\n[CLIP - Zero-Shot Image Classification]")
        report.append(f"  Top-1 Accuracy: {results['clip']['accuracy']*100:.2f}%")
        report.append(f"  Top-5 Accuracy: {results['clip']['top5_accuracy']*100:.2f}%")
        report.append(f"  Best class: {max(results['clip']['per_class_accuracy'].items(), key=lambda x: x[1])}")
    
    if results['lora']:
        report.append("\n[LoRA - Sentiment Classification]")
        report.append(f"  Accuracy: {results['lora']['accuracy']*100:.2f}%")
        report.append(f"  F1-Score: {results['lora']['f1_score']:.4f}")
        report.append(f"  Trainable params: {results['lora']['parameter_stats']['trainable_params']:,}")
        report.append(f"  Efficiency: {results['lora']['parameter_stats']['reduction_factor']:.0f}x reduction")
    
    if results['frozen']:
        report.append("\n[Frozen - Image Captioning]")
        report.append(f"  BLEU-4: {results['frozen']['bleu4']:.4f}")
        report.append(f"  Perplexity: {results['frozen']['perplexity']:.2f}")
        report.append(f"  Trainable %: {results['frozen']['parameter_stats']['trainable_percentage']:.2f}%")
    
    report.append("\n" + "="*70)
    
    # Save report
    report_path = os.path.join(results_base, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"✓ Saved report to {report_path}")
    
    # Print to console
    print("\n" + '\n'.join(report))

def main():
    """Main comparison pipeline"""
    print("="*70)
    print("FOUNDATION MODELS COMPARISON AND VISUALIZATION")
    print("="*70)
    
    # Load results
    results = load_all_results()
    
    if not any(results.values()):
        print("\n❌ No evaluation results found!")
        print("Please run the evaluation scripts first:")
        print("  - python evaluate_clip.py")
        print("  - python evaluate_lora.py")
        print("  - python evaluate_frozen.py")
        return
    
    # Create comparison table
    df = create_comparison_table(results)
    
    # Generate visualizations
    plot_parameter_efficiency(results)
    plot_confusion_matrices(results)
    plot_performance_summary(results)
    
    # Generate report
    generate_report(results, df)
    
    print("\n" + "="*70)
    print("✅ COMPARISON COMPLETE!")
    print("="*70)
    print(f"\nResults saved to:")
    print(f"  - {results_base}/model_comparison.csv")
    print(f"  - {results_base}/evaluation_report.txt")
    print(f"  - {viz_dir}/parameter_efficiency.png")
    print(f"  - {viz_dir}/confusion_matrices.png")
    print(f"  - {viz_dir}/performance_summary.png")
    print("="*70)

if __name__ == "__main__":
    main()
