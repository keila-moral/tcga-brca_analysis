# Methodology

## Phase 1: Image-Based Survival Prediction
**Objective**: Predict patient risk scores directly from Histopathology patches.

 **Method**:
- **Backbone**: ResNet18/50 pretrained on ImageNet.
- **Loss Function**: Cox Proportional Hazards (CoxPH) Loss.
- **Hypothesis**: Morphological features in tumor tissue (nuclear atypia, mitosis) correlate with aggressiveness and survival.

**Papers**:
- *Katzman et al. (2018)* "DeepSurv: Personalized Treatment Recommender System Using A Cox Proportional Hazards Deep Neural Network".
- *Mobadersany et al. (2018)* "Predicting cancer outcomes from histology and genomics using convolutional networks".

## Phase 2: Multimodal Refinement
**Objective**: Improve predictive C-Index by fusing image features with clinical data (stage, age, etc.).

**Method**:
- **Late Fusion**: Concatenate the 512-dim image embedding with encoded clinical vectors.
- **Architecture**: `MultimodalSurvival` network.

**Papers**:
- *Chen et al. (2020)* "Pathomic Fusion: An Integrated Framework for Fusing Histopathology and Genomic Features".

## Phase 3: Generative "Rewinding" (Preventive AI)
**Objective**: Visualizing early-stage disease by translating late-stage images.

**Method**:
- **Model**: UNet-based Generator (Pix2Pix or CycleGAN style).
- **Task**: Train to map $Distribution_{Late} \rightarrow Distribution_{Early}$.
- **Preventive Insight**: By generating "pre-disease" or "early" appearances, we can train clinicians to spot subtle early markers.

**Papers**:
- *Isola et al. (2017)* "Image-to-Image Translation with Conditional Adversarial Networks" (Pix2Pix).
- *Zhu et al. (2017)* "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks" (CycleGAN).
