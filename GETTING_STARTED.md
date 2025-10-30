# Getting Started Guide

## Welcome! 🎉

This project implements three influential papers in foundation models. This guide will help you understand, run, and extend the implementations.

## 📋 Prerequisites

### Knowledge Requirements
- Python programming (intermediate)
- Basic understanding of deep learning
- Familiarity with PyTorch
- Understanding of transformers (helpful but not required)

### Software Requirements
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 16GB RAM minimum

## 🚀 Installation

### Step 1: Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or use conda
conda create -n foundation_models python=3.9
conda activate foundation_models
```

### Step 2: Install Dependencies

```bash
# Install core dependencies
pip install torch torchvision transformers

# Install additional packages
pip install numpy pandas matplotlib seaborn tqdm scikit-learn

# Or install from requirements.txt
cd CLIP_radford21a
pip install -r requirements.txt
```

## 📚 Understanding the Models

### CLIP (Contrastive Language-Image Pre-training)

**What it does**: Learns to match images with text descriptions

**Use cases**:
- Zero-shot image classification
- Image-text retrieval
- Visual search

**Key insight**: Training on 400M image-text pairs enables zero-shot transfer

### LoRA (Low-Rank Adaptation)

**What it does**: Adapts large models with minimal trainable parameters

**Use cases**:
- Fine-tuning large language models
- Task-specific adaptation
- Multi-task model deployment

**Key insight**: Model updates have low intrinsic rank, so can be represented with small matrices

### Frozen

**What it does**: Adds vision to language models without retraining them

**Use cases**:
- Visual question answering
- Image captioning
- Few-shot multimodal tasks

**Key insight**: Train only vision encoder through gradients from frozen language model

## 🔧 Running the Models

### CLIP

#### Basic Usage
```python
from CLIP_radford21a.src.clip_model import CLIP
import torch

# Create model
model = CLIP(embed_dim=512)

# Prepare inputs
images = torch.randn(8, 3, 224, 224)  # Batch of images
texts = ["a dog", "a cat", ...]  # Text descriptions

# Get embeddings
image_features = model.encode_image(images)
# text_features = model.encode_text(input_ids, attention_mask)
```

#### Training
```bash
cd CLIP_radford21a
python src/train.py --epochs 10 --batch_size 128 --lr 1e-4
```

### LoRA

#### Basic Usage
```python
from LoRA_2106.09685v1.src.lora_model import LoRAForLanguageModel
from transformers import GPT2LMHeadModel

# Load base model
base_model = GPT2LMHeadModel.from_pretrained('gpt2')

# Apply LoRA
lora_model = LoRAForLanguageModel(base_model, rank=4, alpha=1)

# Only LoRA parameters are trainable!
trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable:,}")  # ~0.3M vs 125M total
```

#### Training
```bash
cd LoRA_2106.09685v1
python src/train.py --rank 4 --alpha 1 --epochs 3
```

### Frozen

#### Basic Usage
```python
from Frozen_2106.13884v2.src.frozen_model import FrozenModel

# Create model
model = FrozenModel(lm_model_name='gpt2', num_visual_tokens=2)

# Generate caption for image
outputs = model(images=images, input_ids=caption_ids, labels=labels)
loss = outputs.loss
```

#### Training
```bash
cd Frozen_2106.13884v2
python src/train.py --epochs 10 --lr 3e-4
```

## 📊 Comparing Models

### Run Comparison
```bash
cd model_comparison
python compare_models.py
```

This generates:
- Parameter efficiency analysis
- Visualization charts
- Comparison report (JSON)

### Understanding Results

**Parameter Efficiency** (lower is better):
- CLIP: 100% trainable (baseline)
- LoRA: 0.2% trainable (10,000x reduction!)
- Frozen: 0.36% trainable (280x reduction)

**Training Speed** (faster is better):
- LoRA: Fastest (only trains small matrices)
- CLIP: Medium (trains full model)
- Frozen: Slowest (backprop through frozen model)

**Flexibility** (higher is better):
- Frozen: Best (few-shot learning)
- CLIP: Medium (zero-shot transfer)
- LoRA: Good (quick adaptation)

## 🎯 Common Tasks

### Task 1: Zero-Shot Image Classification (CLIP)
```python
class_names = ["dog", "cat", "bird", "car"]
predictions, probs = model.zero_shot_classify(images, class_names)
```

### Task 2: Fine-Tune Model (LoRA)
```python
# Only optimize LoRA parameters
lora_params = [p for n, p in model.named_parameters() if 'lora' in n]
optimizer = torch.optim.AdamW(lora_params, lr=3e-4)
```

### Task 3: Visual Question Answering (Frozen)
```python
# Provide few examples then ask question
prompt = "Q: What color is the car? A: Blue\nQ: What is this? A:"
model.generate(images=image, prompt_text=prompt)
```

## 🐛 Troubleshooting

### Issue: Out of Memory

**Solution**:
```python
# Reduce batch size
batch_size = 16  # Instead of 128

# Use gradient accumulation
accumulation_steps = 8

# Enable mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### Issue: Slow Training

**Solution**:
- Use GPU instead of CPU
- Enable DataParallel for multi-GPU
- Reduce image resolution
- Use fewer transformer layers

### Issue: Poor Performance

**Solution**:
- Train for more epochs
- Increase model size
- Use more data
- Adjust learning rate
- Check data preprocessing

## 📖 Next Steps

### Beginner
1. Run provided examples
2. Understand model architectures
3. Experiment with hyperparameters
4. Visualize embeddings

### Intermediate
1. Train on custom datasets
2. Implement evaluation metrics
3. Optimize training pipeline
4. Compare with baselines

### Advanced
1. Combine multiple methods (CLIP + LoRA)
2. Implement variants (Vision Transformer, etc.)
3. Scale to larger models
4. Contribute improvements

## 📚 Learning Resources

### Papers
1. CLIP: https://arxiv.org/abs/2103.00020
2. LoRA: https://arxiv.org/abs/2106.09685
3. Frozen: https://arxiv.org/abs/2106.13884

### Tutorials
- PyTorch official tutorials
- Hugging Face transformers docs
- Papers with Code

### Communities
- PyTorch forums
- Hugging Face forums
- ML subreddits

## 💡 Tips & Tricks

1. **Start small**: Use small models and datasets first
2. **Monitor training**: Use TensorBoard or Wandb
3. **Save checkpoints**: Regularly save model states
4. **Document experiments**: Keep track of what works
5. **Ask questions**: Join ML communities

## 🤝 Contributing

Want to improve the project? Areas to contribute:
- Bug fixes
- Documentation improvements
- New features
- Optimization
- Additional baselines

## 📞 Support

Having issues? 
- Check documentation
- Review code comments
- Open an issue
- Ask in discussions

---

**Happy coding! 🚀**

Remember: These are research implementations. Start simple, experiment often, and have fun learning!
