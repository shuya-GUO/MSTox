# MSTox

MSTox is a multi-task learning framework for unified acute toxicity prediction across multiple species and exposure routes based on mass spectrum fingerprints.

The framework integrates species-specific and exposure-route information into a multimodal deep learning architecture and enables simultaneous prediction of LD50 values for multiple biological conditions.

---

## Overview

MSTox is designed for multi-species and multi-route acute toxicity prediction using 5000-bit mass spectrum fingerprints.

The framework supports:

- Multiple species:
  - Mouse
  - Rat
  - Rabbit
  - Dog
  - Cat
  - Guinea pig

- Multiple exposure routes:
  - Oral
  - Intravenous(iv)

The model predicts:

- Log-transformed LD50 values
- Original LD50 values (mg/kg)
- Toxicity categories

---

## Framework Architecture

MSTox is built upon a Cross-Attention-based multimodal deep learning framework consisting of:

- Chemical Encoder
- Condition Encoder
- Cross-Attention Layer
- Multi-gate Mixture-of-Experts (MMoE)
- Adaptive Task Heads

### Chemical Encoder

The chemical encoder uses a two-layer fully connected neural network to transform 5000-dimensional mass spectrum fingerprints into 128-dimensional chemical feature vectors.

### Condition Encoder

The condition encoder integrates:

- Species embedding (32 dimensions)
- Exposure route embedding (8 dimensions)
- Outer-product interaction features (256 dimensions)

These features are projected into a 128-dimensional condition representation.

### Feature Fusion

Chemical and condition features are fused through a Cross-Attention mechanism to capture interactions between molecular structures and biological conditions.

### MMoE Module

The fused representations are refined using a Multi-gate Mixture-of-Experts (MMoE) module for task-specific feature extraction.

### Graph Regularization

Task embeddings are mapped into feature space to construct endpoint relationship graphs for graph regularization and enhanced task correlation learning.

### Adaptive Task Heads

Adaptive task heads dynamically determine learning strategies based on:

- Low-performance tasks:
  - R² < 0.5

- Extremely low-sample tasks:
  - Sample size < 20

Different strategies include:

- Residual shrinkage
- Ordinal classification
- Specialized prediction heads

---

## Input

### Molecular Representation

- 5000-bit mass spectrum fingerprints

---

## Output

### Regression Tasks

- Log-transformed LD50 values
- Original LD50 values (mg/kg)

### Toxicity Classification

| Toxicity Level | LD50 Range |
|---|---|
| High Toxicity | LD50 < 100 mg/kg |
| Moderate Toxicity | 100 mg/kg ≤ LD50 ≤ 1000 mg/kg |
| Low Toxicity | LD50 > 1000 mg/kg |

---

## Installation

### Conda Environment

```bash
conda env create -f MSTox.yml
conda activate MSTox
```

### Pip Installation

```bash
pip install -r requirements.txt
```

---

## Project Structure

```text
MSTox/
│
├── data/                                # Dataset
│   ├──all_species_route_summary.csv     # Mass spectrum fingerprints for training
│   ├── ECRFS_fps.csv                    # Mass spectrum fingerprints for testing
│   ├── Sharing High-Quality LC-MS_MS Spectral Data of over 100 Emerging Chemical Risks in the Food Chain_1_all.zip                 # ECRFS dataset metadata involving MS/MS information
│   ├── mouse/                           # Mouse MS & LD50 data (oral & iv)
│   ├── rat/                             # Rat MS & LD50 data (oral & iv)
│   ├── rabbit/                          # Rabbit MS & LD50 data (oral & iv)
│   ├── dog/                             # Dog MS & LD50 data (oral & iv)
│   ├── cat/                             # Cat MS & LD50 data (oral & iv)
│   └── guinea_pig/                      # Guinea pig MS & LD50 data (oral & iv)
├── notebooks/                           # Data processing notebooks
├── train.ipynb                          # Training notebook
├── predict.py                           # Inference script
├── requirements.txt
├── MSTox.yml
├── LICENSE
└── README.md
```

> **Note:** Trained model weights (`models/`) are not included due to file size.
> Please download them separately and place them in the `models/` directory before running `predict.py`.

---

## Usage

### Training

```bash
jupyter notebook train.ipynb
```

Or execute all cells headlessly:

```bash
jupyter nbconvert --to notebook --execute train.ipynb
```

### Prediction

```bash
python predict.py
```

Make sure the trained model weights are placed in the `models/` directory before running inference.

---

## Dependencies

Main dependencies include:

- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Optuna
- RDKit

---

## Toxicity Tasks

MSTox supports unified toxicity prediction for combinations of:

- Species
- Exposure routes

under a multi-task learning paradigm.

---

## Future Work

Potential future extensions include:

- Additional species and exposure routes
- Pretraining strategies for low-resource toxicity tasks
- Explainable AI for toxicity mechanism interpretation

---

## License

MIT License

---

## Citation

If you use MSTox in your research, please cite our paper:

```bibtex
@article{mstox2026,
  title={Predicting Cross-Species and Cross-Route Acute Toxicity from Mass Spectra via a Multi-Modal Deep Learning Framework},
  author={Shuya Guo, Qiao Xue,* Jianjie Fu*},
  year={2026}，
  note = {Manuscript in preparation}
}
```
