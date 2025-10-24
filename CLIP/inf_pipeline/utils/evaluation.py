import torch
import clip
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

def get_text_features(model, class_names, prompt_templates, device):
    """
    Creates and normalizes text features, ensembling if multiple templates are provided.
    """
    with torch.no_grad():
        all_template_features = []
        for template in prompt_templates:
            # Create text descriptions for all classes
            text_descriptions = [template.format(c.replace("_", " ")) for c in class_names]
            text_inputs = clip.tokenize(text_descriptions).to(device)
            
            template_features = model.encode_text(text_inputs)
            template_features /= template_features.norm(dim=-1, keepdim=True)
            all_template_features.append(template_features)
            
        # Average across all templates
        text_features = torch.stack(all_template_features).mean(dim=0)
        # Re-normalize after averaging
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
    return text_features

def evaluate_model(model, dataset_loader, class_names, prompt_templates, device):
    """
    Runs zero-shot inference for a given model and dataset loader.
    Returns a results dictionary, and raw predictions/labels for analysis.
    """
    print("Building text features...")
    text_features = get_text_features(model, class_names, prompt_templates, device)
    
    all_preds = []
    all_labels = []
    all_top5_correct = []

    print(f"Running inference on {len(dataset_loader.dataset)} images...")
    with torch.no_grad():
        for images, labels in tqdm(dataset_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Get image features
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Calculate similarity
            # (batch_size, num_classes)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # --- Get Top-1 and Top-5 Predictions ---
            _, top1_preds = similarity.max(dim=1)
            _ , top5_preds = similarity.topk(5, dim=1)
            
            # Check if true label is in top 5
            labels_expanded = labels.view(-1, 1)
            top5_correct = (top5_preds == labels_expanded).any(dim=1)
            
            all_preds.append(top1_preds)
            all_labels.append(labels)
            all_top5_correct.append(top5_correct)

    # Combine all results from all batches
    y_pred = torch.cat(all_preds).cpu().numpy()
    y_true = torch.cat(all_labels).cpu().numpy()
    y_top5_correct = torch.cat(all_top5_correct).cpu().numpy()

    # --- Generate Metrics ---
    top1_accuracy = accuracy_score(y_true, y_pred)
    top5_accuracy = np.mean(y_top5_correct)
    
    # Get full classification report (precision, recall, f1)
    class_report_dict = classification_report(
        y_true, y_pred, target_names=class_names, digits=3, output_dict=True
    )
    
    print(f"Evaluation Complete. Top-1: {top1_accuracy*100:.2f}%, Top-5: {top5_accuracy*100:.2f}%")

    # Package all results
    results = {
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "full_classification_report": class_report_dict,
        "class_names": class_names
    }
    
    return results, y_true, y_pred
