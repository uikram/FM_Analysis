import os
import torch
from torchvision.datasets import CIFAR10, CIFAR100, Caltech101, ImageFolder
from torch.utils.data import DataLoader, Subset

# Import the helper lists
from imagenet_helper.imagenet_classes import CLASSES as IMAGENET_CLASSES
from imagenet_helper.imagenet_templates import TEMPLATES as IMAGENET_TEMPLATES

def load_dataset(dataset_name, dataset_config, preprocess, limit_images=None):
    """
    Loads a specified dataset, class names, and prompt templates.
    """
    root = os.path.expanduser(dataset_config['root_path'])
    
    # --- 1. Load Dataset ---
    # --- FIXED: Changed all checks to be lowercase to match config.yml ---
    if dataset_name == "cifar10":
        dataset = CIFAR10(root, download=True, train=False, transform=preprocess)
        class_names = dataset.classes
        prompt_templates = ["a photo of a {}."]
        
    elif dataset_name == "cifar100":
        dataset = CIFAR100(root, download=True, train=False, transform=preprocess)
        class_names = dataset.classes
        prompt_templates = ["a photo of a {}."]
        
    elif dataset_name == "caltech101":
        dataset = Caltech101(root, download=True, transform=preprocess)
        # Filter out the background class, which is standard for Caltech-101 eval
        valid_indices = [i for i, (img, label) in enumerate(dataset) if dataset.categories[label] != "BACKGROUND_Google"]
        dataset = Subset(dataset, valid_indices)
        
        class_names = [c for c in dataset.dataset.categories if c != "BACKGROUND_Google"]
        class_names.sort() # Ensure consistent order
        
        # We need to remap labels, this is a bit complex
        # Create a mapping from old label (index in .categories) to new label (index in class_names)
        old_to_new_label_map = {old_idx: new_idx for new_idx, cat in enumerate(class_names) 
                              for old_idx, old_cat in enumerate(dataset.dataset.categories) if old_cat == cat}

        # Subclass the Subset to remap labels on the fly
        class RemappedSubset(Subset):
            def __getitem__(self, idx):
                image, old_label = super().__getitem__(idx)
                new_label = old_to_new_label_map.get(old_label)
                # Handle cases where the label might not be in our map (e.g., BACKGROUND_Google was filtered)
                if new_label is None:
                    # This shouldn't happen with the valid_indices filter, but as a safeguard:
                    return self.__getitem__((idx + 1) % len(self)) 
                return image, new_label

        dataset = RemappedSubset(dataset.dataset, dataset.indices)
        prompt_templates = ["a photo of a {}."]
        
    elif dataset_name == "imagenet":
        val_path = dataset_config['imagenet_val_path']
        if val_path is None or not os.path.exists(os.path.expanduser(val_path)):
            print(f"Warning: ImageNet path not found or not set in config.yml: {val_path}")
            print("Skipping ImageNet evaluation.")
            return None, None, None
            
        dataset = ImageFolder(os.path.expanduser(val_path), transform=preprocess)
        class_names = IMAGENET_CLASSES
        prompt_templates = IMAGENET_TEMPLATES
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # --- 2. Apply Image Limit (for testing) ---
    if limit_images is not None:
        indices = torch.randperm(len(dataset))[:limit_images].tolist()
        dataset = Subset(dataset, indices)
        print(f"--- NOTE: Limiting dataset to {limit_images} random images for a smoke test. ---")

    # --- 3. Create DataLoader ---
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Successfully loaded dataset: {dataset_name}. Num images: {len(dataset)}. Num classes: {len(class_names)}.")
    return loader, class_names, prompt_templates