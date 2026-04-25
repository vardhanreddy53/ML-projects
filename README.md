# ML Projects — Sai Vardhan Reddy Pathuri

A collection of machine learning projects implemented in PyTorch and HuggingFace Transformers, covering computer vision, sequence modeling, and vision-language models.

---

## Projects

### 1. [CLIP Zero-Shot Classification & Retrieval (CIFAR-100)](./clip_zero_shot/)
Zero-shot image classification and semantic text-to-image retrieval using OpenAI's CLIP (ViT-B/32) — no fine-tuning, no labeled training data.

| Method | Top-1 Accuracy |
|---|---|
| Bare class names | 91.11% |
| Prompt-engineered templates | 92.50% |

Text-to-image retrieval achieved perfect top-5 precision across all 4 queries.

**Stack:** PyTorch · HuggingFace Transformers · CIFAR-100 · scikit-learn

---

### 2. [GRU & LSTM Activity Recognition (HAPT Dataset)](./gru_lstm_hapt/)
Sequence classification on raw inertial sensor data with systematic ablations across hidden size, depth, and architecture type.

| Model | Hidden Size | Layers | Test Accuracy |
|---|---|---|---|
| GRU | 32 | 1 | 84.25% |
| GRU | 128 | 1 | **91.99%** |
| GRU | 32 | 2 | 88.30% |
| GRU | 32 | 3 | 87.55% |
| GRU | 32 | 4 | 90.00% |
| LSTM | 32 | 1 | 76.50% |

**Stack:** PyTorch · UCI HAPT Dataset

---

### 3. [CNN on CIFAR-10](./cnn_cifar10/)
Custom CNN architecture with hyperparameter sweeps (learning rate, batch size, optimizer) and architecture modifications (Dropout, BatchNorm).

| Configuration | Test Accuracy |
|---|---|
| Base CNN (lr=0.001, Adam) | 74.82% |
| + Dropout (p=0.5) | 78.14% |
| + BatchNorm + Dropout | **79.92%** |

**Stack:** PyTorch · torchvision · CIFAR-10

---

## Setup

```bash
git clone https://github.com/vardhanreddy53/ML-projects.git
cd ML-projects
pip install torch torchvision transformers scikit-learn matplotlib seaborn
```

Each project is a self-contained Jupyter notebook. Open and run cells sequentially.

---

## Skills Demonstrated

- Vision-language models (CLIP, zero-shot learning, prompt engineering)
- Recurrent neural networks (GRU, LSTM, ablation studies)
- CNNs (architecture design, regularization, hyperparameter tuning)
- PyTorch (custom training loops, DataLoaders, model evaluation)
- HuggingFace Transformers
