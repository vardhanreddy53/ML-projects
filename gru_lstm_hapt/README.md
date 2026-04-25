# Human Activity Recognition with GRU & LSTM

Sequence classification on the UCI HAPT dataset using GRU and LSTM networks in PyTorch — with systematic ablations across hidden size, depth, and architecture type.

---

## Overview

Human Activity and Postural Transitions (HAPT) is a classic time-series classification problem using raw inertial sensor data (accelerometer + gyroscope) from smartphones. This project implements and benchmarks recurrent neural network classifiers, exploring how architecture choices affect accuracy.

---

## Dataset

- **Source:** [UCI HAPT Dataset](https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions)
- **Input:** 6 sensor channels (3-axis accelerometer + 3-axis gyroscope)
- **Classes:** 12 activity/transition labels (walking, sitting, standing, transitions, etc.)
- **Format:** Raw time-series sequences, pre-split into train/test

---

## Architecture

### Baseline GRU
- Input size: 6
- Hidden size: 32
- Layers: 1
- Output: Linear → 12-class softmax
- Loss: CrossEntropyLoss
- Optimizer: Adam
- Epochs: 10

---

## Ablation Studies

| Experiment | Variable | Values Tested |
|---|---|---|
| Hidden Size | `hidden_size` | 32, 128 |
| Network Depth | `num_layers` | 1, 2, 3, 4 |
| Architecture | Model type | GRU vs LSTM |

Each configuration was trained independently and evaluated on the held-out test set.

---

## Results

| Model | Hidden Size | Layers | Test Accuracy |
|---|---|---|---|
| GRU | 32 | 1 | 84.25% |
| GRU | 128 | 1 | 91.99% |
| GRU | 32 | 2 | 88.30% |
| GRU | 32 | 3 | 87.55% |
| GRU | 32 | 4 | 90.00% |
| LSTM | 32 | 1 | 76.50% |

**Key takeaway:** Increasing hidden size (32→128) had the largest single impact (+7.74%), while GRU consistently outperformed LSTM at equivalent configuration (84.25% vs 76.50%).

---

## Tech Stack

- `Python 3.x`
- `PyTorch`
- `NumPy`
- `scikit-learn`
- `matplotlib`

---

## Project Structure
## Project Structure

```
ML-Projects/
└── CLIP_Zero-Shot_Classification_&_Retrieval_(CIFAR-100).ipynb
```

---
The notebook is self-contained and covers all experiments sequentially:

Data loading, windowing, and DataLoader setup
GRU and LSTM model definitions
Training loop with per-epoch loss logging
All ablation runs with a final summary results table

## How to Run

```bash
# Install dependencies
pip install torch numpy scikit-learn matplotlib

Download the HAPT dataset from the UCI repository
Update DATA_DIR in the notebook to point to your local dataset path
Run the notebook top to bottom
```

---
'''
## Concepts Demonstrated

- **Recurrent neural networks** — GRU and LSTM for variable-length sequences
- **Time-series classification** — raw inertial sensor data, no hand-crafted features
- **Ablation studies** — isolating the effect of hidden size, depth, and architecture
- **PyTorch best practices** — custom Dataset, DataLoader, training loop, model checkpointing

---

## References

- [HAPT Dataset — UCI ML Repository](https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions)
- [GRU Paper — Cho et al., 2014](https://arxiv.org/abs/1406.1078)
- [LSTM Paper — Hochreiter & Schmidhuber, 1997](https://www.bioinf.jku.at/publications/older/2604.pdf)
