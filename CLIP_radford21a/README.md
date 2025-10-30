# CLIP Implementation

Implementation of "Learning Transferable Visual Models From Natural Language Supervision" 
(Radford et al., 2021, [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)).

## Files

- `src/clip_model.py` - Model architecture
- `src/train.py` - Standard training script
- `src/train_a5000.py` - A5000-optimized training
- `requirements.txt`: Required Python packages.

## Setup

1.  Create and activate a virtual environment (e.g., using conda or venv).
2.  Install the dependencies from the project root:
    ```bash
    pip install -r CLIP_radford21a/requirements.txt
    ```

## Usage

Navigate to the `src` directory to run the training scripts.

```bash
cd CLIP_radford21a/src
```

**Standard Training (CPU or default GPU):**
```bash
python train.py --epochs 10 --batch_size 32
```

**A5000 optimized (Targeting GPU #2):**
```bash
CUDA_VISIBLE_DEVICES=2 python train_a5000.py --batch_size 64
```
