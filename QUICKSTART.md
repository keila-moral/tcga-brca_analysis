# Quick Start Guide

This guide will help you get started with the TCGA-BRCA GenAI project quickly.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended, but CPU also works)
- 16GB+ RAM
- ~100GB disk space for data

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/tcga-brca-genai.git
cd tcga-brca-genai
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Basic installation
pip install -r requirements.txt

# Or install as package
pip install -e .

# For development
pip install -e ".[dev]"

# For all features
pip install -e ".[medical,generative,logging]"
```

## Data Preparation

### Option 1: Using Your Own Data

If you have TCGA-BRCA data:

```bash
# 1. Place raw images in data/raw/images/
mkdir -p data/raw/images
cp /path/to/your/images/* data/raw/images/

# 2. Place clinical data CSV in data/raw/
cp /path/to/clinical.csv data/raw/clinical.csv

# 3. Preprocess images
python src/data/preprocessing.py \
    --input-dir data/raw/images \
    --output-dir data/processed/images \
    --patch-size 256 \
    --normalize-stain \
    --quality-check
```

### Option 2: Download from GDC (TCGA Data Portal)

```bash
# Install GDC Data Transfer Tool
pip install gdc-client

# Download data (you'll need a manifest file from GDC portal)
gdc-client download -m manifest.txt -d data/raw/
```

## Running Your First Experiment

### Phase 1: Image-Based Survival Prediction

```bash
# 1. Check your data
python -c "from src.data.dataset import TCGAImageSurvivalDataset; \
           dataset = TCGAImageSurvivalDataset('data/processed/images', 'data/raw/clinical.csv'); \
           print(f'Dataset size: {len(dataset)}')"

# 2. Train model
python src/train/train_survival.py \
    --config configs/survival_config.yaml \
    --wandb  # Optional: enable logging

# 3. Monitor training
# If using wandb, check: https://wandb.ai/your-username/tcga-brca-genai
# Otherwise, check results/models/phase1_survival/

# 4. View results
ls results/models/phase1_survival/
# You should see:
# - checkpoint_best.pt (best model)
# - checkpoint_latest.pt (latest checkpoint)
# - training_curves.png (loss and C-index plots)
```

### Testing Your Installation

Run unit tests to verify everything is working:

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_dataset.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Project Workflow

### Complete Workflow Example

```bash
# 1. Preprocess data
python src/data/preprocessing.py \
    --input-dir data/raw/images \
    --output-dir data/processed/images

# 2. Phase 1: Train survival model
python src/train/train_survival.py \
    --config configs/survival_config.yaml

# 3. Phase 2: Train multimodal model
python src/train/train_multimodal.py \
    --config configs/multimodal_config.yaml \
    --pretrained-model results/models/phase1_survival/checkpoint_best.pt

# 4. Phase 3: Train generative model
python src/train/train_generative.py \
    --config configs/generative_config.yaml
```

## Common Issues and Solutions

### Issue: Out of Memory Error

**Solution 1**: Reduce batch size
```yaml
# In configs/survival_config.yaml
training:
  batch_size: 16  # Reduce from 32
```

**Solution 2**: Use gradient accumulation
```python
# In training script
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Issue: CUDA Not Available

**Check CUDA installation**:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Solution**: Train on CPU (slower)
```yaml
# In config file
device: "cpu"
```

### Issue: Data Loading Slow

**Solution 1**: Increase num_workers
```yaml
data:
  num_workers: 8  # Increase from 4
```

**Solution 2**: Enable image caching (if you have enough RAM)
```yaml
data:
  cache_images: true
```

### Issue: Import Errors

**Make sure you're in the right directory**:
```bash
# Should be in project root
pwd
# Should show: /path/to/tcga-brca-genai

# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Next Steps

### 1. Explore Jupyter Notebooks

```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

### 2. Customize Configurations

Edit `configs/survival_config.yaml` to experiment with:
- Different architectures (ResNet vs ViT)
- Learning rates
- Augmentation strategies
- Model capacity

### 3. Read Documentation

- `docs/methodology.md` - Detailed methodology explanation
- `docs/model_selection.md` - Why we chose specific models
- `docs/references.md` - All relevant papers and citations

### 4. Join the Community

- Open issues for bugs or questions
- Submit pull requests for improvements
- Share your results and insights

## Monitoring Training

### Using Weights & Biases

```bash
# Login to wandb
wandb login

# Train with logging
python src/train/train_survival.py \
    --config configs/survival_config.yaml \
    --wandb

# View at: https://wandb.ai/your-username/tcga-brca-genai
```

### Using TensorBoard

```bash
# Start tensorboard
tensorboard --logdir results/logs/

# View at: http://localhost:6006
```

## Minimal Example

If you just want to test the code without full data:

```python
# Create dummy data
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

# Create directories
Path('data/test/images').mkdir(parents=True, exist_ok=True)

# Create dummy images
for i in range(10):
    img = Image.new('RGB', (256, 256), color=(i*25, i*25, i*25))
    img.save(f'data/test/images/patient_{i:03d}.png')

# Create dummy clinical data
clinical = pd.DataFrame({
    'patient_id': [f'patient_{i:03d}' for i in range(10)],
    'survival_time': np.random.rand(10) * 2000,
    'vital_status': np.random.choice(['Alive', 'Dead'], 10)
})
clinical.to_csv('data/test/clinical.csv', index=False)

# Test dataset
from src.data.dataset import TCGAImageSurvivalDataset
dataset = TCGAImageSurvivalDataset(
    'data/test/images',
    'data/test/clinical.csv'
)
print(f"Dataset size: {len(dataset)}")
print(f"Sample: {dataset[0].keys()}")
```

## Getting Help

- **Issues**: Open a GitHub issue with your question
- **Documentation**: Check `docs/` directory
- **Examples**: See `notebooks/` for working examples
- **Tests**: Run `pytest tests/` to verify functionality

## Performance Benchmarks

Expected training times (NVIDIA V100):

| Phase | Model | Epochs | Time |
|-------|-------|--------|------|
| Phase 1 | ResNet50 | 50 | ~2-3 hours |
| Phase 2 | Multimodal | 30 | ~1-2 hours |
| Phase 3 | CycleGAN | 100 | ~4-6 hours |

Expected performance (C-index):

- Image-only baseline: 0.65-0.70
- Multimodal: 0.70-0.75
- State-of-the-art: 0.75-0.80

## Citation

If you use this code, please cite:

```bibtex
@software{tcga_brca_genai,
  title={TCGA-BRCA Multi-Modal Survival Prediction and Generative AI},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/tcga-brca-genai}
}
```

---

**Happy coding! 🚀**

For questions or issues, please open a GitHub issue or contact the maintainers.
