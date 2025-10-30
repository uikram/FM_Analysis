#!/usr/bin/env python3
"""
Master launch script to run training for CLIP, LoRA, and Frozen models.
Places this script in the root directory (e.g., FM_Analysis/) alongside
the model project folders (CLIP_radford21a, LoRA_..., etc.).

Usage:
  python launch.py all
  python launch.py clip
  python launch.py lora --rank 4
"""
import os
import sys
import subprocess
import argparse

# Set the GPU for all subprocesses, as specified in the guide
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def check_gpu():
    """Checks if PyTorch can see the GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ Found GPU: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("✗ Error: PyTorch imported but torch.cuda.is_available() is False.")
            print("  Please check your PyTorch/CUDA installation.")
            return False
    except ImportError:
        print("✗ Error: PyTorch is not installed. Please install it.")
        return False
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        return False

def run_command(cmd, cwd):
    """Runs a shell command in a specified directory."""
    print("\n" + "="*60)
    print(f"RUNNING: '{cmd}' in './{cwd}'")
    print("="*60)
    
    # shell=True is used for simplicity, as in the PDF
    process = subprocess.run(cmd, shell=True, cwd=cwd)
    
    print("-" * 60)
    if process.returncode == 0:
        print(f"✓ SUCCESS: Command finished for '{cwd}'")
    else:
        print(f"✗ FAILED: Command for '{cwd}' (Exit code: {process.returncode})")
    print("="*60)
    return process.returncode == 0

def train_clip(args):
    """Runs the A5000-optimized CLIP training."""
    cmd = f"python src/train_a5000.py --epochs {args.epochs} --batch_size {args.batch_size_clip}"
    return run_command(cmd, cwd="CLIP_radford21a")

def train_lora(args):
    """Runs the A5000-optimized LoRA training."""
    cmd = f"python src/train_a5000.py --epochs {args.epochs} --batch_size {args.batch_size_lora} --rank {args.rank}"
    return run_command(cmd, cwd="LoRA_2106.09685v1")

def train_frozen(args):
    """Runs the A5000-optimized Frozen training."""
    cmd = f"python src/train_a5000.py --epochs {args.epochs} --batch_size {args.batch_size_frozen}"
    return run_command(cmd, cwd="Frozen_2106.13884v2")

def main():
    parser = argparse.ArgumentParser(
        description="Master training script for Foundation Models.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'model', 
        choices=['clip', 'lora', 'frozen', 'all'], 
        help="The model to train, or 'all' to train all three sequentially."
    )
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs to train (default: 10)")
    parser.add_argument('--rank', type=int, default=8, help="Rank for LoRA (default: 8)")
    
    # Add model-specific batch sizes based on the PDF's A5000 optimizations
    parser.add_argument('--batch_size_clip', type=int, default=64, help="Batch size for CLIP (default: 64)")
    parser.add_argument('--batch_size_lora', type=int, default=32, help="Batch size for LoRA (default: 32)")
    parser.add_argument('--batch_size_frozen', type=int, default=16, help="Batch size for Frozen (default: 16)")
    
    args = parser.parse_args()

    print("Checking for GPU...")
    if not check_gpu():
        print("Exiting due to GPU issue.")
        return 1
    
    print(f"\nAttempting to run: {args.model.upper()}")
    
    success_clip = True
    success_lora = True
    success_frozen = True
    
    if args.model == 'clip' or args.model == 'all':
        success_clip = train_clip(args)
        
    if args.model == 'lora' or args.model == 'all':
        success_lora = train_lora(args)
        
    if args.model == 'frozen' or args.model == 'all':
        success_frozen = train_frozen(args)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    all_successful = success_clip and success_lora and success_frozen
    
    if args.model == 'all':
        print(f"CLIP:    {'✓ SUCCESS' if success_clip else '✗ FAILED'}")
        print(f"LoRA:    {'✓ SUCCESS' if success_lora else '✗ FAILED'}")
        print(f"Frozen:  {'✓ SUCCESS' if success_frozen else '✗ FAILED'}")
    else:
        print(f"{args.model.upper()}: {'✓ SUCCESS' if all_successful else '✗ FAILED'}")

    print("="*60)

    return 0 if all_successful else 1

if __name__ == "__main__":
    sys.exit(main())
