import os
import torch
import clip
from PIL import Image
import numpy as np
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- 1. Set up your specific GPU (from your code) ---
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
print(f"CUDA_VISIBLE_DEVICES set to: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# --- 2. Check the device ---
if torch.cuda.is_available():
    print(f"PyTorch sees GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda") 
else:
    print("CUDA is not available. Using CPU.")
    device = torch.device("cpu")

def run_zero_shot_analysis(model_name, dataset, class_names, prompt_template):
    """
    Loads a model, runs zero-shot inference on a dataset, and returns metrics.
    """
    print(f"\n--- Loading Model: {model_name} ---")
    
    # Load the model and its specific preprocessor
    model, preprocess = clip.load(model_name, device=device)
    
    # We must re-load the dataset with the *specific* preprocess for this model
    # (e.g., ViT-L/14@336px needs a 336px image, not 224px)
    print("Applying model-specific preprocessing to dataset...")
    test_dataset = CIFAR10(root=os.path.expanduser("~/.cache"), 
                           download=True, 
                           train=False, 
                           transform=preprocess)
    
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    # --- Prepare Text Prompts ---
    print("Tokenizing text prompts...")
    text_descriptions = [prompt_template.format(c) for c in class_names]
    text_inputs = clip.tokenize(text_descriptions).to(device)

    all_preds = []
    all_labels = []

    print(f"Running inference on {len(test_dataset)} images...")
    with torch.no_grad():
        # Get text features once
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Get image features
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Calculate similarity
            # (batch_size, num_classes)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # Get predictions
            _, preds = similarity.max(dim=1)
            
            all_preds.append(preds)
            all_labels.append(labels)

    # Combine all results
    all_preds_tensor = torch.cat(all_preds).cpu().numpy()
    all_labels_tensor = torch.cat(all_labels).cpu().numpy()

    return all_preds_tensor, all_labels_tensor

def generate_report_and_plots(model_name, y_true, y_pred, class_names):
    """
    Generates a classification report and saves a confusion matrix plot.
    """
    safe_model_name = model_name.replace('/', '_') # For valid filenames
    
    # --- 1. Classification Report ---
    print(f"\n--- Classification Report for {model_name} ---")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    print(report)
    
    # Save report to a text file
    with open(f"report_{safe_model_name}.txt", "w") as f:
        f.write(f"Classification Report for {model_name}\n")
        f.write(report)
        
    # --- 2. Confusion Matrix ---
    print("Generating Confusion Matrix plot...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{safe_model_name}.png")
    plt.close()
    
    # --- 3. Calculate Overall Accuracy ---
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy for {model_name}: {accuracy*100:.2f}%")
    return accuracy

def plot_summary_results(results_dict):
    """
    Generates a bar chart comparing the accuracy of all models.
    """
    print("\n--- Generating Final Accuracy Comparison Plot ---")
    df = pd.DataFrame(list(results_dict.items()), columns=['Model', 'Accuracy'])
    df = df.sort_values(by='Accuracy', ascending=False)
    
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='Model', y='Accuracy', data=df)
    ax.set_title('CLIP Model Zero-Shot Accuracy on CIFAR-10', fontsize=18)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_xlabel('Model', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Add labels to the bars
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}%", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points')
    
    plt.ylim(0, max(df['Accuracy']) * 1.1) # Add some space at the top
    plt.tight_layout()
    plt.savefig("final_accuracy_comparison.png")
    plt.close()
    print("All analysis complete. See 'final_accuracy_comparison.png' for summary.")


def main():
    # --- List of all models from the paper available in the library ---
    # The 5 ResNets and 4 ViT variants (including the high-res one)
    models_to_test = [
        "RN50",
        # "RN101",
        # "RN50x4",
        # "RN50x16",
        # "RN50x64",
        # "ViT-B/32",
        # "ViT-B/16",
        # "ViT-L/14",
        # "ViT-L/14@336px"
    ]

    # --- Load CIFAR-10 Dataset Info ---
    # We only load it once to get class names.
    # The actual data will be loaded and preprocessed *per model*
    print("Loading CIFAR-10 class names...")
    cifar10 = CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False)
    class_names = cifar10.classes
    print(f"CIFAR-10 Classes: {class_names}")

    # --- Prompt Engineering ---
    # Using a good prompt is crucial for high accuracy
    prompt_template = "a photo of a {}"
    print(f"Using prompt template: '{prompt_template}'")

    model_accuracies = {}

    for model_name in models_to_test:
        # 1. Run inference
        y_pred, y_true = run_zero_shot_analysis(model_name, cifar10, class_names, prompt_template)
        
        # 2. Generate reports and plots
        accuracy = generate_report_and_plots(model_name, y_true, y_pred, class_names)
        
        # 3. Store final accuracy
        model_accuracies[model_name] = accuracy * 100

    # 4. Generate final summary plot
    plot_summary_results(model_accuracies)

if __name__ == "__main__":
    main()
