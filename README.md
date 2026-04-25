# Ligand–Receptor Binding Prediction (ML Project)

## Overview
This project predicts whether a molecule binds using molecular descriptors extracted from SDF data.

## Workflow
- Load SDF file using RDKit
- Extract molecular features (MolWt, LogP, HDonor, HAcceptor)
- Create labels using threshold
- Train Random Forest model
- Evaluate performance

## Tech Stack
Python, RDKit, Scikit-learn, Pandas

## Results
- Accuracy, Classification Report, Confusion Matrix, ROC-AUC

## Note
This project uses synthetic labeling for demonstration due to lack of real binding annotations.
