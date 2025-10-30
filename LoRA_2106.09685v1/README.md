# LoRA Implementation

This directory contains a Python implementation of Low-Rank Adaptation (LoRA) 
from the paper "LoRA: Low-Rank Adaptation of Large Language Models" 
(Hu et al., 2021, [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)).

## Files

- `src/lora_model.py`: Contains the core LoRA logic, including `LoRALayer` and `LoRALinear` classes, and a helper function `apply_lora_to_model` to patch a model.
- `src/train.py`: A standard training script using a dummy dataset to demonstrate LoRA fine-tuning.
- `src/train_a5000.py`: A training script with minor optimizations (e.g., more data workers) intended for a more powerful GPU like an RTX A5000.
- `requirements.txt`: Required Python packages.

## Setup

1.  Create and activate a virtual environment (e.g., using conda or venv).
2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Navigate to the `src` directory to run the training scripts.

```bash
cd LoRA_2106.09685v1/src
```

**Standard Training (CPU or default GPU):**

This script will use `cuda:0` if available, otherwise CPU.

```bash
python train.py --rank 4 --epochs 3
```

**Optimized Training (Targeting a specific GPU, e.g., GPU #2):**

This command tells the script to only see the system's 2nd GPU, which the script will then access as `cuda:0`.

```bash
CUDA_VISIBLE_DEVICES=2 python train_a5000.py --rank 8 --epochs 3
```

Checkpoints containing only the LoRA weights will be saved in the `LoRA_2106.09685v1/checkpoints/` directory.
