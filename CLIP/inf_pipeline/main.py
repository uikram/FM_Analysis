import os
import yaml
import clip
import torch
import time

# Import our custom modules
import utils.data
import utils.evaluation
import utils.data_analysis

def load_config(config_path='config.yml'):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    start_time = time.time()
    config = load_config()

    # --- 1. Setup Environment ---
    settings = config['settings']
    results_dir = os.path.expanduser(settings['results_dir'])
    os.environ["CUDA_VISIBLE_DEVICES"] = settings['cuda_device']
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")

    # This list will store the high-level results for the final summary plot
    all_results_summary = []

    # --- 2. Outer Loop: Models ---
    for model_name in config['models']:
        print(f"\n{'='*80}\nProcessing Model: {model_name}\n{'='*80}")
        try:
            model, preprocess = clip.load(model_name, device=device)
        except Exception as e:
            print(f"Failed to load model {model_name}. Skipping. Error: {e}")
            continue

        # --- 3. Inner Loop: Datasets ---
        for dataset_key, dataset_config in config['datasets'].items():
            print(f"\n--- Processing Dataset: {dataset_key} ---")
            
            # --- 4. Load Data ---
            loader, class_names, prompts = utils.data.load_dataset(
                dataset_key, 
                dataset_config, 
                preprocess,
                limit_images=settings.get('limit_images')
            )
            
            if loader is None:
                continue # Skip if dataset (e.g., ImageNet) is not found
            
            # --- 5. Run Evaluation ---
            # evaluation.py now returns the full results_dict, plus y_true/y_pred for the confusion matrix
            results_dict, y_true, y_pred = utils.evaluation.evaluate_model(
                model, loader, class_names, prompts, device
            )
            
            # Add metadata to the results
            results_dict['model_name'] = model_name
            results_dict['dataset_name'] = dataset_key
            
            # --- 6. Save All Artifacts ---
            print("--- Saving artifacts ---")
            
            # Save the full JSON report
            utils.data_analysis.save_json_report(
                results_dict, model_name, dataset_key, results_dir
            )
            
            # --- THIS IS THE FIX ---
            # Save the human-readable .txt report
            # This call now correctly passes the *entire* results dictionary
            utils.data_analysis.save_classification_report_txt(
                results_dict, model_name, dataset_key, results_dir
            )
            
            # Save the confusion matrix plot
            utils.data_analysis.save_confusion_matrix_plot(
                y_true, y_pred, class_names, model_name, dataset_key, results_dir
            )
            
            # Append high-level metrics for the final summary graph
            all_results_summary.append({
                "model": model_name,
                "dataset": dataset_key,
                "Top-1 Accuracy": results_dict['top1_accuracy'] * 100,
                "Top-5 Accuracy": results_dict['top5_accuracy'] * 100
            })

    # --- 7. Generate Final Summary Graphs ---
    if all_results_summary:
        utils.data_analysis.generate_summary_graph(all_results_summary, results_dir)

    end_time = time.time()
    print(f"\n{'='*80}\nTotal Analysis Finished in: {(end_time - start_time) / 60:.2f} minutes")
    print(f"All results saved in: {os.path.abspath(results_dir)}")

if __name__ == "__main__":
    main()