# TCGA-BRCA Multi-Modal Survival Prediction & Generative AI

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

This project implements a multi-modal machine learning and generative AI workflow for breast cancer (TCGA-BRCA) analysis with two primary objectives:

1. **Survival Prediction**: Calculate survival chances and predict patient outcomes using histopathology images and clinical data
2. **Temporal Image Generation**: Generate early-stage or pre-disease histopathology images for preventive analysis

## 🏗️ Architecture

The project follows a three-phase approach:

### Phase 1: Image-Based Survival Prediction
- Use whole slide images (WSI) or image patches from TCGA-BRCA dataset
- Train deep learning models (ResNet50/ViT) for survival analysis
- Implement Cox Proportional Hazards model for time-to-event prediction

### Phase 2: Multimodal Fusion
- Integrate clinical/tabular data (symptoms, biomarkers, demographics)
- Fuse image features with clinical embeddings
- Refine survival predictions through multimodal learning

### Phase 3: Generative Disease Progression Modeling
- Train conditional GANs or diffusion models
- Generate synthetic images showing disease at earlier stages
- Enable preventive analysis and early intervention strategies

## 📁 Project Structure

```
tcga-brca-genai/
├── data/
│   ├── raw/              # Original TCGA-BRCA data (immutable)
│   ├── processed/        # Preprocessed tensors and images
│   └── interim/          # Intermediate processing outputs
├── src/
│   ├── data/
│   │   ├── download_tcga.py      # TCGA data acquisition
│   │   ├── dataset.py            # PyTorch Dataset classes
│   │   └── preprocessing.py      # Image preprocessing pipeline
│   ├── models/
│   │   ├── survival_net.py       # Survival prediction models
│   │   ├── fusion_net.py         # Multimodal fusion architecture
│   │   └── generative.py         # GAN/Diffusion models
│   ├── train/
│   │   ├── train_survival.py     # Phase 1 training script
│   │   ├── train_multimodal.py   # Phase 2 training script
│   │   └── train_generative.py   # Phase 3 training script
│   └── utils/
│       ├── metrics.py            # Survival metrics (C-index, etc.)
│       ├── visualization.py      # Plotting and logging
│       └── config.py             # Configuration management
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_image_preprocessing.ipynb
│   └── 03_results_visualization.ipynb
├── tests/
│   ├── test_dataset.py
│   ├── test_models.py
│   └── test_metrics.py
├── docs/
│   ├── methodology.md            # Detailed methodology
│   ├── references.md             # Citations and papers
│   └── model_selection.md        # Model choice rationale
├── configs/
│   ├── survival_config.yaml
│   ├── multimodal_config.yaml
│   └── generative_config.yaml
├── results/
│   ├── models/                   # Saved model checkpoints
│   ├── figures/                  # Generated plots
│   └── logs/                     # Training logs
├── requirements.txt
├── setup.py
└── .gitignore
```

## 🚀 Installation & Quick Start

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM
- **No manual data downloads required!** ✨

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tcga-brca-genai.git
cd tcga-brca-genai
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. **Start training immediately with GDC API** (no downloads needed!):
```bash
# Train with automatic data fetching from GDC
python src/train/train_survival_gdc.py \
    --config configs/gdc_streaming_config.yaml \
    --n-samples 50
```

That's it! The script automatically:
- ✅ Fetches clinical data from GDC servers
- ✅ Downloads images on-demand during training
- ✅ Caches everything for future use
- ✅ Handles all data matching and preprocessing

## 📊 Usage

### Quick Start with GDC API (Recommended)

```bash
# Option 1: Streaming mode - download images on-demand (fastest to start)
python src/train/train_survival_gdc.py \
    --config configs/gdc_streaming_config.yaml \
    --n-samples 100

# Option 2: Download-first mode - download all images before training
python src/train/train_survival_gdc.py \
    --config configs/gdc_streaming_config.yaml \
    --n-samples 100 \
    --download-first

# Option 3: Programmatic access to GDC data
python -c "
from src.data.gdc_client import GDCDataManager
manager = GDCDataManager()
info = manager.setup_dataset(n_samples=50)
df = manager.get_matched_data()
print(df.head())
"
```

See [docs/gdc_access_guide.md](docs/gdc_access_guide.md) for complete GDC usage guide.

### Traditional Workflow (Manual Data)

If you have pre-downloaded TCGA-BRCA data:

### Phase 1: Image-Based Survival Prediction

```bash
# Preprocess images
python src/data/preprocessing.py \
    --input-dir data/raw \
    --output-dir data/processed \
    --image-size 256

# Train survival model
python src/train/train_survival.py \
    --config configs/survival_config.yaml \
    --data-dir data/processed \
    --output-dir results/models
```

### Phase 2: Multimodal Fusion

```bash
# Train multimodal model
python src/train/train_multimodal.py \
    --config configs/multimodal_config.yaml \
    --image-features results/models/survival_features.pt \
    --clinical-data data/raw/clinical.csv
```

### Phase 3: Generative Image Translation

```bash
# Train generative model
python src/train/train_generative.py \
    --config configs/generative_config.yaml \
    --data-dir data/processed \
    --checkpoint results/models/generative_model.pt
```

## 🧪 Testing

Run automated tests:
```bash
pytest tests/
```

Run specific test modules:
```bash
pytest tests/test_dataset.py -v
pytest tests/test_models.py -v
```

## 📈 Evaluation Metrics

### Survival Prediction
- **C-index (Concordance Index)**: Primary metric for survival model evaluation
- **Integrated Brier Score**: Time-dependent prediction accuracy
- **Calibration plots**: Assess prediction reliability

### Generative Models
- **Fréchet Inception Distance (FID)**: Image quality metric
- **Structural Similarity Index (SSIM)**: Image similarity
- **Expert evaluation**: Pathologist assessment of generated images

## 📚 Key References

See [docs/references.md](docs/references.md) for complete bibliography. Key papers include:

1. **Survival Analysis**: Katzman et al. (2018) - DeepSurv
2. **Histopathology Deep Learning**: Campanella et al. (2019) - Clinical-grade computational pathology
3. **Generative Models**: Isola et al. (2017) - Pix2Pix for image-to-image translation
4. **Medical Image Generation**: Uzunova et al. (2019) - Generative models for medical images

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- The Cancer Genome Atlas (TCGA) for providing the BRCA dataset
- PyTorch and MONAI communities for excellent tools
- Research community for foundational work in computational pathology

## 📧 Contact

For questions or collaborations, please open an issue or contact [your-email@example.com].

---

**Note**: This project is for research purposes only and should not be used for clinical decision-making without proper validation and regulatory approval.
