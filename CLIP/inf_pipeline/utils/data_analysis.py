import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def save_json_report(results, model_name, dataset_name, output_dir):
    """Saves the full results dictionary as a JSON file."""
    safe_model_name = model_name.replace('/', '_')
    report_dir = os.path.join(output_dir, "json_reports", safe_model_name)
    os.makedirs(report_dir, exist_ok=True)
    
    filepath = os.path.join(report_dir, f"{dataset_name}.json")
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved JSON report to: {filepath}")

def save_classification_report_txt(results, model_name, dataset_name, output_dir):
    """Saves the human-readable classification report as a .txt file."""
    safe_model_name = model_name.replace('/', '_')
    # --- THIS IS THE FIX ---
    # Changed report_.dir to report_dir
    report_dir = os.path.join(output_dir, "classification_reports", safe_model_name)
    os.makedirs(report_dir, exist_ok=True)
    
    filepath = os.path.join(report_dir, f"{dataset_name}.txt")
    
    # Re-create the string version of the report
    report_str = f"--- Classification Report for {model_name} on {dataset_name} ---\n\n"
    report_str += f"Top-1 Accuracy: {results['top1_accuracy']*100:.2f}%\n"
    report_str += f"Top-5 Accuracy: {results['top5_accuracy']*100:.2f}%\n\n"
    
    headers = ["", "precision", "recall", "f1-score", "support"]
    report_str += f"{headers[0]:<20} {headers[1]:>10} {headers[2]:>10} {headers[3]:>10} {headers[4]:>10}\n"
    report_str += "-"*65 + "\n"
    
    report_dict = results["full_classification_report"]
    class_names = results["class_names"]
    
    # Handle potentially incomplete reports (e.g., if a class had 0 predictions)
    for i, class_name in enumerate(class_names):
        class_str = str(i)
        if class_name in report_dict:
             metrics = report_dict[class_name]
        elif class_str in report_dict:
             metrics = report_dict[class_str]
        else:
            # This can happen if a class had 0 samples in a limited run
            print(f"Warning: No metrics found for class '{class_name}' (index {i}). Skipping in report.")
            continue
            
        report_str += (f"{class_name:<20} "
                       f"{metrics['precision']:>10.3f} "
                       f"{metrics['recall']:>10.3f} "
                       f"{metrics['f1-score']:>10.3f} "
                       f"{metrics['support']:>10.0f}\n")
    
    report_str += "\n" + "-"*65 + "\n"
    
    for avg_key in ['accuracy', 'macro avg', 'weighted avg']:
        if avg_key not in report_dict:
            continue
            
        metrics = report_dict[avg_key]
        if avg_key == 'accuracy':
            # Accuracy is a single float in the report_dict
             report_str += f"{avg_key:<20} {'':>10} {'':>10} {metrics:>10.3f} {report_dict['macro avg']['support']:>10.0f}\n"
        else:
            report_str += (f"{avg_key:<20} "
                           f"{metrics['precision']:>10.3f} "
                           f"{metrics['recall']:>10.3f} "
                           f"{metrics['f1-score']:>10.3f} "
                           f"{metrics['support']:>10.0f}\n")

    with open(filepath, 'w') as f:
        f.write(report_str)
    print(f"Saved .txt report to: {filepath}")


def save_confusion_matrix_plot(y_true, y_pred, class_names, model_name, dataset_name, output_dir):
    """Generates and saves a confusion matrix plot."""
    safe_model_name = model_name.replace('/', '_')
    plot_dir = os.path.join(output_dir, "confusion_matrices", safe_model_name)
    os.makedirs(plot_dir, exist_ok=True)
    filepath = os.path.join(plot_dir, f"{dataset_name}.png")

    cm = confusion_matrix(y_true, y_pred)
    
    # Plotting is slow for large class numbers (like ImageNet)
    if len(class_names) > 150:
        print("Skipping confusion matrix plot for large dataset (ImageNet).")
        return

    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name} on {dataset_name}', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Reduce font size and rotate for many classes (e.g., Caltech-101)
    if len(class_names) > 40:
        plt.xticks(rotation=90, fontsize=6)
        plt.yticks(rotation=0, fontsize=6)
    else:
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"Saved confusion matrix to: {filepath}")

def generate_summary_graph(all_results_summary, output_dir):
    """
    Generates a grouped bar chart for all models and datasets.
    """
    print("\n--- Generating Final Accuracy Comparison Plots ---")
    graphs_dir = os.path.join(output_dir, "accuracy_graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    
    if not all_results_summary:
        print("No results to plot. Exiting.")
        return

    df = pd.DataFrame(all_results_summary)
    
    # Plot 1: Grouped by Model
    for dataset_name in df['dataset'].unique():
        plt.figure(figsize=(16, 9))
        df_dataset = df[df['dataset'] == dataset_name]
        
        df_melted = df_dataset.melt('model', var_name='Metric', value_name='Accuracy', 
                                    value_vars=['Top-1 Accuracy', 'Top-5 Accuracy'])
        
        ax = sns.barplot(x='model', y='Accuracy', hue='Metric', data=df_melted)
        ax.set_title(f'CLIP Model Zero-Shot Accuracy on {dataset_name}', fontsize=18)
        ax.set_ylabel('Accuracy (%)', fontsize=14)
        ax.set_xlabel('Model', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}%", 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', xytext=(0, 9), textcoords='offset points')
        
        plt.ylim(0, 100)
        plt.legend(loc='upper left', fontsize=12)
        plt.tight_layout()
        graph_path = os.path.join(graphs_dir, f"summary_accuracy_{dataset_name}.png")
        plt.savefig(graph_path)
        plt.close()
        print(f"Saved summary graph to: {graph_path}")