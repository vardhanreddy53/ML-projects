# Zero-Shot Image Classification & Text-to-Image Retrieval with CLIP

A deep dive into OpenAI's CLIP (ViT-B/32) for zero-shot classification and semantic image retrieval on a 12-class subset of CIFAR-100 — no fine-tuning, no task-specific training.

---

## Overview

This project explores two capabilities of CLIP's joint vision-language embedding space:

1. **Zero-Shot Classification** — classify images using only natural language prompts, without any labeled training data
2. **Text-to-Image Retrieval** — rank images by semantic similarity to a free-form text query

---

## Dataset

- **Source:** CIFAR-100 (via `torchvision`)
- **Subset:** 12 classes × 30 images = 360 images total
- **Classes:** Selected to test both easy and visually ambiguous categories

---

## Tasks

### 1. Baseline Zero-Shot Classification
- Encoded bare class-name prompts (e.g., `"bicycle"`, `"forest"`) as text embeddings
- Classified each image by cosine similarity between image and text embeddings
- Computed per-class and overall top-1 accuracy

### 2. Prompt Engineering
- Designed **4+ prompt templates per class** (e.g., `"a photo of a {class}"`, `"a close-up of a {class} in the wild"`)
- Averaged normalized embeddings across templates to create robust class prototypes
- Compared accuracy vs. bare class-name baseline
- Plotted confusion matrix to identify misclassification patterns

### 3. Text-to-Image Retrieval
- Encoded 4 natural language queries (e.g., `"a yellow school bus"`, `"a person skiing on snow"`)
- Ranked all 360 images by cosine similarity to each query
- Displayed **top-5 retrieved images** per query

---

## Key Results

| Method | Top-1 Accuracy |
|---|---|
| Bare class names | 91.11% |
| Prompt-engineered templates | 92.50% |
| Improvement | +1.39% |

### Text-to-Image Retrieval Queries
- `"a yellow school bus"` → Top-5 all correctly retrieved (bus class)
- <img width="2144" height="466" alt="retrieval_a_yellow_school" src="https://github.com/user-attachments/assets/32d92980-0741-4c17-b19f-5672e476a37f" />

- `"a flying butterfly"` → Top-5 all correctly retrieved (butterfly class)
- <img width="2144" height="466" alt="retrieval_a_flying_butter" src="https://github.com/user-attachments/assets/69b6a3ec-beb1-4c20-8933-2003f507a75d" />

- `"a desk lamp"` → Top-5 all correctly retrieved (lamp class)
- <img width="2144" height="466" alt="retrieval_a_desk_lamp" src="https://github.com/user-attachments/assets/4c17b8de-2626-4646-9512-8e33845d3e76" />

- `"a turtle swimming"` → Top-5 all correctly retrieved (turtle class)
- <img width="2144" height="466" alt="retrieval_a_turtle_swimmi" src="https://github.com/user-attachments/assets/cf03b170-52bb-47d3-a6bb-d86e24baca6c" />

---

## Tech Stack

- `Python 3.x`
- `PyTorch`
- `HuggingFace CLIP` (`openai/clip-vit-base-patch32`)
- `torchvision`
- `scikit-learn`
- `matplotlib` / `seaborn`

---

## Project Structure

```
ML-Projects/
└── CLIP_Zero-Shot_Classification_&_Retrieval_(CIFAR-100).ipynb
```

---

## How to Run

```bash
# Install dependencies
pip install torch torchvision transformers scikit-learn matplotlib pillow

Then open and run the notebook top to bottom. CIFAR-100 downloads automatically on first run.
```

---

## Concepts Demonstrated

- **Vision-language models** — understanding joint embedding spaces
- **Zero-shot generalization** — no task-specific fine-tuning or labeled data
- **Prompt engineering** — improving model performance through better text inputs
- **Semantic retrieval** — cosine similarity over normalized embeddings
- **Evaluation** — confusion matrices, per-class accuracy analysis

---

## References

- [CLIP Paper — Radford et al., 2021](https://arxiv.org/abs/2103.00020)
- [HuggingFace CLIP Docs](https://huggingface.co/openai/clip-vit-base-patch32)
- [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
