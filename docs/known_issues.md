# Known issues

## Issues common to several phases:

 **Dataset**

- **Limited Image Dataset Size**: As a prototype, the primary challenge is the extremely small dataset used for training and evaluation. Patient data downloads take significant time; while API access was considered as an alternative, connectivity problems led to preferring a reduced local dataset for this initial project phase.

- **Class Imbalance**: The tabular dataset indicates all patients are alive, so analysis should prioritize overall_survival_months over vital_status.

- **Missing tumor_stage**: Nested diagnoses arrays in the tabular dataset were mishandled during retrieval and excluded. Update query_clinical_data() in src/data/download_gdc_metadata.py to fix this

 **Model validation**

- **Suboptimal Metrics**: For Phases 1 and 2, where the goal is predicting patient survival rates, metrics like the Concordance index (C-index) or Kaplan-Meier median survival time (K-M median) would better classify low- vs. high-risk patients. The current Cox PH loss struggles with minibatches and fails to yield a unique value per patient in its implementation. 


## Issues relative to specific phases:

### Phase 1

- **Model Architecture**:

### Phase 2

- **Model Architecture**:

### Phase 3

- **Generator vs. Discriminator imbalance**:


