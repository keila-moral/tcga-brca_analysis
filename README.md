# TCGA-BRCA ML+Gen AI Project

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This project builds a comprehensive ML + Generative AI workflow using the TCGA-BRCA (Breast Invasive Carcinoma) dataset. The goal is to:
1.  **Predict Survival**: Use Histopathology images (WSIs/Patches) to predict patient outcomes.
2.  **Multimodal Integration**: Refine predictions by integrating clinical tabular data.
3.  **Preventive Gen AI**: Generate "early-stage" disease images from late-stage samples to visualize disease progression "in reverse".

## Project Structure
```
tcga-brca-genai/
├── data/
│   ├── raw/          # Original data (not committed)
│   ├── processed/    # Preprocessed patches and tensors
├── src/
│   ├── data/         # Data loaders and preprocessing
│   ├── models/       # Deep Learning architectures
│   ├── train/        # Training and validation loops
│   └── utils/        # Helper functions
├── notebooks/        # Experiments and analysis
├── docs/             # Methodology and citations
└── requirements.txt
```

## Setup
1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  (Optional) Install OpenSlide binaries if on Mac:
    ```bash
    brew install openslide
    ```

## Data
Data is sourced from the GDC Portal (TCGA-BRCA).
- **Images**: Whole Slide Images (WSIs) formatted as `.svs`.
- **Clinical**: Tabular data with survival information.
*Note: This project uses ROI patches extracted from WSIs for computational efficiency.*

## Methodology
(To be updated as the project progresses)
- **Phase 1**: Image-based survival modeling (CNN/ViT).
- **Phase 2**: Clinical + Image Fusion.
- **Phase 3**: Generative adversarial/diffusion models for stage translation.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- The Cancer Genome Atlas (TCGA) for providing the BRCA dataset
- PyTorch and MONAI communities for excellent tools
- Research community for foundational work in computational pathology

## 📧 Contact

For questions or collaborations, please open an issue or contact [kmoralfig@gmail.com].

---

**Note**: This project is for research purposes only and should not be used for clinical decision-making without proper validation and regulatory approval.
