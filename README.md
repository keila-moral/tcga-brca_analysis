# TCGA-BRCA ML+Gen AI Project

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
