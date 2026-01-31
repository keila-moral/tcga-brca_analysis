#!/bin/bash

# Activate venv
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.

echo "Processing Pipeline Started..."
echo "Waiting for image patches to appear in data/processed/patches..."

# Wait loop
while [ -z "$(ls -A data/processed/patches 2>/dev/null)" ]; do
   sleep 30
   echo "Waiting for data..."
done

echo "Data detected! Waiting an extra minute to ensure initial batch is ready..."
sleep 60

echo "----------------------------------------------------------------"
echo "Phase 1: Starting Survival Prediction Training (Baseline)"
echo "----------------------------------------------------------------"
python -u src/train/train_survival.py

echo "----------------------------------------------------------------"
echo "Phase 2: Starting Multimodal Fusion Training"
echo "----------------------------------------------------------------"
python -u src/train/train_fusion.py

echo "----------------------------------------------------------------"
echo "Phase 3: Starting Generative Model Training (Preventive AI)"
echo "----------------------------------------------------------------"
python -u src/train/train_gen.py

echo "----------------------------------------------------------------"
echo "Pipeline Complete."
echo "----------------------------------------------------------------"
