# CNN on CIFAR-10

Custom CNN architecture with hyperparameter sweeps and progressive architecture improvements on the CIFAR-10 image classification benchmark.

---

## Architecture

3× Conv → LeakyReLU → AvgPool blocks, followed by fully-connected layers.

- **Loss:** CrossEntropyLoss
- **Optimizer:** Adam (baseline)
- **Epochs:** 20

---

## Experiments

### (i) Baseline
| Config | Train Acc | Test Acc |
|---|---|---|
| lr=0.001, batch=64, Adam | 98.18% | 74.82% |

### (ii) Hyperparameter Sweeps

**Learning Rate** (batch=64, Adam):
| LR | Test Acc |
|---|---|
| 0.01 | ~66% |
| 0.001 | 74.82% |
| 0.0001 | 69.44% |

**Batch Size** (lr=0.001, Adam):
| Batch Size | Test Acc |
|---|---|
| 32 | ~75% |
| 64 | 74.82% |
| 128 | 75.85% |

**Optimizer** (lr=0.001, batch=64):
| Optimizer | Test Acc |
|---|---|
| Adam | 74.82% |
| SGD + momentum | ~74% |
| AdamW | 75.40% |

### (iii) Architecture Modifications
| Model | Train Acc | Test Acc |
|---|---|---|
| Base CNN | 98.18% | 74.82% |
| + Dropout (p=0.5) | 93.65% | 78.14% |
| + BatchNorm + Dropout | 94.66% | **79.92%** |

**Key insight:** Base CNN significantly overfits (98% train vs 74% test). Dropout reduced overfitting; adding BatchNorm further improved generalization to ~80% test accuracy.

---

## Tech Stack

- `Python 3.x`
- `PyTorch`
- `torchvision` (CIFAR-10)
- `matplotlib`

---

## How to Run

```bash
pip install torch torchvision matplotlib
jupyter notebook CNN_on_CIFAR-10.ipynb
```

---

## Concepts Demonstrated

- Custom CNN design with LeakyReLU and AvgPool
- Hyperparameter sweeps (learning rate, batch size, optimizer)
- Regularization techniques: Dropout and BatchNorm
- Overfitting diagnosis and mitigation
- Training/test accuracy curve analysis
