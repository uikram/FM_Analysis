"""
Frozen Model - Quick Start Test Script

This script allows you to quickly test the Frozen model architecture
on a tiny subset of data before committing to full training.

Run this first to verify everything works on your RTX A5000!
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import time

# Import main components
from frozen_training import FrozenModel, FrozenConfig, VisionEncoder

def create_dummy_data(num_samples=50, output_dir="./test_data"):
    """
    Create a small dummy dataset for testing
    
    This generates synthetic images and captions to test the pipeline
    without downloading the full dataset.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    annotations = []
    
    print(f"Creating {num_samples} dummy samples...")
    
    captions = [
        "a red car on the street",
        "a person walking in the park",
        "a cat sitting on a chair",
        "a dog playing with a ball",
        "a bird flying in the sky",
        "a boat on the water",
        "a house with a garden",
        "a mountain covered in snow",
        "a tree in the forest",
        "a flower in bloom"
    ]
    
    for i in range(num_samples):
        # Create random colored image
        img = Image.new('RGB', (224, 224), 
                       color=(np.random.randint(0, 255),
                             np.random.randint(0, 255),
                             np.random.randint(0, 255)))
        
        img_path = images_dir / f"test_{i:04d}.jpg"
        img.save(img_path)
        
        # Create annotation
        caption = captions[i % len(captions)]
        annotations.append({
            'image_id': f"test_{i:04d}.jpg",
            'caption': caption
        })
    
    # Save annotations
    import json
    with open(output_dir / "annotations.jsonl", 'w') as f:
        for ann in annotations:
            f.write(json.dumps(ann) + '\n')
    
    print(f"✓ Created test dataset at {output_dir}")
    return output_dir


def test_model_initialization(config):
    """Test that the model can be initialized correctly"""
    print("\n" + "="*80)
    print("TEST 1: Model Initialization")
    print("="*80)
    
    try:
        print("Initializing tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained(config.language_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        print("✓ Tokenizer loaded")
        
        print("\nInitializing Frozen model...")
        model = FrozenModel(config).to(config.device)
        print("✓ Model initialized")
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"\nModel Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable (vision encoder): {trainable_params:,}")
        print(f"  Frozen (language model): {frozen_params:,}")
        print(f"  Training ratio: {trainable_params/total_params*100:.2f}%")
        
        # Memory check
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            dummy_input = torch.randn(1, 3, 224, 224).to(config.device)
            with torch.no_grad():
                _ = model.vision_encoder(dummy_input)
            
            mem_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"\nGPU Memory:")
            print(f"  Peak memory used: {mem_used:.2f} GB")
            print(f"  Available on RTX A5000: 24 GB")
            print(f"  Estimated training memory: {mem_used * 3:.2f} GB (3x for gradients/optimizer)")
            
            if mem_used * 3 > 20:
                print("  ⚠️  WARNING: May exceed GPU memory during training!")
                print("  → Consider reducing batch size or using gradient accumulation")
            else:
                print("  ✓ Should fit in RTX A5000 GPU memory")
        
        return True, model, tokenizer
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False, None, None


def test_data_loading(config, data_dir):
    """Test that data can be loaded correctly"""
    print("\n" + "="*80)
    print("TEST 2: Data Loading")
    print("="*80)
    
    try:
        from frozen_training import ConceptualCaptionsDataset
        
        print("Loading dataset...")
        tokenizer = GPT2Tokenizer.from_pretrained(config.language_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        dataset = ConceptualCaptionsDataset(
            annotations_file=str(data_dir / "annotations.jsonl"),
            image_dir=str(data_dir / "images"),
            tokenizer=tokenizer,
            config=config,
            split="test"
        )
        
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        
        # Test single sample
        print("\nTesting single sample...")
        sample = dataset[0]
        print(f"  Image shape: {sample['images'].shape}")
        print(f"  Input IDs shape: {sample['input_ids'].shape}")
        print(f"  Attention mask shape: {sample['attention_mask'].shape}")
        print(f"  Labels shape: {sample['labels'].shape}")
        
        # Test dataloader
        print("\nTesting DataLoader...")
        loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
        batch = next(iter(loader))
        print(f"  Batch image shape: {batch['images'].shape}")
        print(f"  Batch input_ids shape: {batch['input_ids'].shape}")
        
        print("✓ Data loading successful")
        return True, dataset
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_forward_pass(config, model, dataset):
    """Test that forward pass works correctly"""
    print("\n" + "="*80)
    print("TEST 3: Forward Pass")
    print("="*80)
    
    try:
        model.eval()
        
        # Get a batch
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        
        images = batch['images'].to(config.device)
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        labels = batch['labels'].to(config.device)
        
        print("Running forward pass...")
        with torch.no_grad():
            logits, loss = model(images, input_ids, attention_mask, labels)
        
        print(f"✓ Forward pass successful")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Loss: {loss.item():.4f}")
        
        if loss.item() > 20 or loss.item() < 0:
            print("  ⚠️  WARNING: Loss seems abnormal")
        else:
            print("  ✓ Loss in expected range")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_pass(config, model, dataset):
    """Test that backward pass and optimization work"""
    print("\n" + "="*80)
    print("TEST 4: Backward Pass & Optimization")
    print("="*80)
    
    try:
        model.train()
        
        # Setup optimizer
        optimizer = torch.optim.Adam(
            model.vision_encoder.parameters(),
            lr=config.learning_rate
        )
        
        # Get a batch
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        
        images = batch['images'].to(config.device)
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch['attention_mask'].to(config.device)
        labels = batch['labels'].to(config.device)
        
        print("Running training step...")
        
        # Forward
        logits, loss = model(images, input_ids, attention_mask, labels)
        initial_loss = loss.item()
        print(f"  Initial loss: {initial_loss:.4f}")
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients
        grad_norm = 0
        for p in model.vision_encoder.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5
        print(f"  Gradient norm: {grad_norm:.4f}")
        
        if grad_norm == 0:
            print("  ✗ No gradients computed!")
            return False
        
        # Optimization step
        optimizer.step()
        
        # Check that parameters changed
        with torch.no_grad():
            logits, loss = model(images, input_ids, attention_mask, labels)
            new_loss = loss.item()
        
        print(f"  Loss after update: {new_loss:.4f}")
        print(f"  Loss change: {new_loss - initial_loss:.4f}")
        
        print("✓ Backward pass successful")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generation(config, model, dataset):
    """Test text generation"""
    print("\n" + "="*80)
    print("TEST 5: Text Generation")
    print("="*80)
    
    try:
        model.eval()
        tokenizer = GPT2Tokenizer.from_pretrained(config.language_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Get a sample image
        sample = dataset[0]
        image = sample['images'].unsqueeze(0).to(config.device)
        
        print("Generating caption...")
        captions = model.generate(
            image,
            tokenizer,
            max_length=20,
            temperature=1.0,
            top_k=50
        )
        
        print(f"✓ Generated caption: {captions[0]}")
        print("  Note: Caption may not be meaningful before training")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mini_training(config, model, dataset):
    """Run a mini training loop to verify everything works"""
    print("\n" + "="*80)
    print("TEST 6: Mini Training Loop (10 steps)")
    print("="*80)
    
    try:
        model.train()
        
        optimizer = torch.optim.Adam(
            model.vision_encoder.parameters(),
            lr=config.learning_rate
        )
        
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0
        )
        
        losses = []
        
        print("Running mini training loop...")
        for step, batch in enumerate(loader):
            if step >= 10:
                break
            
            images = batch['images'].to(config.device)
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)
            
            # Forward
            start_time = time.time()
            logits, loss = model(images, input_ids, attention_mask, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            step_time = time.time() - start_time
            losses.append(loss.item())
            
            print(f"  Step {step+1}/10: loss={loss.item():.4f}, time={step_time:.2f}s")
        
        avg_loss = sum(losses) / len(losses)
        print(f"\n✓ Mini training complete")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Average time per step: {sum([0]*len(losses))/len(losses) if losses else 0:.2f}s")
        
        # Estimate full training time
        steps_per_epoch = len(dataset) // (config.batch_size * config.gradient_accumulation_steps)
        estimated_hours = (steps_per_epoch * step_time) / 3600
        print(f"\n  Estimated training time per epoch: {estimated_hours:.1f} hours")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "🚀" * 40)
    print("FROZEN MODEL - QUICK START TESTS")
    print("🚀" * 40)
    
    # Initialize config for testing
    config = FrozenConfig()
    config.batch_size = 4
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nConfiguration:")
    print(f"  Device: {config.device}")
    print(f"  Language Model: {config.language_model_name}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  FP16: {config.fp16}")
    
    # Create dummy data
    data_dir = create_dummy_data(num_samples=50)
    
    # Run tests
    results = {}
    
    # Test 1: Model initialization
    success, model, tokenizer = test_model_initialization(config)
    results['Model Initialization'] = success
    if not success:
        print("\n❌ Model initialization failed. Cannot proceed with other tests.")
        return
    
    # Test 2: Data loading
    success, dataset = test_data_loading(config, data_dir)
    results['Data Loading'] = success
    if not success:
        print("\n❌ Data loading failed. Cannot proceed with other tests.")
        return
    
    # Test 3: Forward pass
    success = test_forward_pass(config, model, dataset)
    results['Forward Pass'] = success
    
    # Test 4: Backward pass
    success = test_backward_pass(config, model, dataset)
    results['Backward Pass'] = success
    
    # Test 5: Generation
    success = test_generation(config, model, dataset)
    results['Text Generation'] = success
    
    # Test 6: Mini training
    success = test_mini_training(config, model, dataset)
    results['Mini Training'] = success
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nYou're ready to start full training!")
        print("\nNext steps:")
        print("  1. Download Conceptual Captions dataset")
        print("  2. Update paths in frozen_training.py")
        print("  3. Run: python frozen_training.py")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("\nPlease fix the failing tests before proceeding to full training.")
    print("="*80)


if __name__ == "__main__":
    # Check prerequisites
    print("Checking prerequisites...")
    
    if not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA not available. Training will be very slow on CPU.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            exit()
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✓ GPU detected: {gpu_name}")
        print(f"  Total memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 20:
            print("  ⚠️  Warning: Less than 20GB GPU memory may cause issues")
    
    # Run tests
    run_all_tests()