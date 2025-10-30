# Frozen Implementation

Multimodal Few-Shot Learning with Frozen Language Models (Tsimpoukelli et al., 2021)

## Files

- `src/frozen_model.py` - Frozen model with vision encoder
- `src/train.py` - Standard training script
- `src/train_a5000.py` - A5000-optimized training

## Usage

```bash
python src/train.py --epochs 10 --batch_size 16
CUDA_VISIBLE_DEVICES=2 python src/train_a5000.py --batch_size 16
```
