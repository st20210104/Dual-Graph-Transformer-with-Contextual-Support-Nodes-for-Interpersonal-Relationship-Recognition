# Dual-Graph Transformer with Contextual Support Nodes for Interpersonal Relationship Recognition

This repository contains an implementation of the **Dual-Graph Transformer architecture with Contextual Support Nodes (CSNs)** for **Interpersonal Relationship Recognition (IPR)** from images.  
The approach is based on the methodology from the paper:

> **Dual-Graph Transformer with Contextual Support Nodes for Interpersonal Relationship Recognition**  
> Simge Akay, Duygu Cakir, Nafiz Arica – *IEEE IPTA 2025*  

---

## 📄 Overview

The method models two complementary graphs for each person pair in an image:

1. **Pair-Centric Graph** (Local CSN): Captures fine-grained interpersonal cues such as age gap, gaze coupling, posture similarity.
2. **Scene-Centric Graph** (Visual + Textual CSNs): Encodes macro context including scene elements and language-based semantic cues from captions.

A **Gated Graph Neural Network (GGNN)** processes each graph separately, and their embeddings are fused via a **Transformer encoder** for relationship classification.

---

## 🏗 Architecture
- **Nodes**:
  - Pair-Centric Graph: Person A, Person B, Local CSN
  - Scene-Centric Graph: Person A, Person B, Visual CSN, Textual CSN
- **Edges**: Fully connected between person nodes and CSNs.
- **Fusion**: Transformer-based attention for inter-graph and inter-modality reasoning.

---

## 📂 Repository Structure

```
dual_graph_ipr/
├── main.py                # Training and evaluation loop
├── models.py              # Dual-Graph Transformer model with GGNN and Transformer fusion
├── utils.py               # Graph creation, dataset handling, evaluation metrics
├── feature_extraction.py  # Visual and textual feature extraction (ResNet, BLIP, BERT)
└── README.md              # This file
```

---

## 🔧 Installation

```bash
git clone https://github.com/username/dual-graph-ipr.git
cd dual-graph-ipr

# (Optional) create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision dgl transformers scikit-learn pillow
```

---

## 🚀 Usage

### Train & Evaluate
```bash
python main.py
```

### Modify Hyperparameters
In `main.py`:
```python
NUM_CLASSES = 6
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 50
HIDDEN_SIZE = 256
N_STEPS = 3
```

---

## 📊 Evaluation Metrics
The training script computes:
- **Accuracy**
- **Per-class Accuracy**
- **Mean Average Precision (mAP)**

Example output:
```
Epoch [1/50] Loss: 1.7325 | Acc: 0.4000 | mAP: 0.2550
Per-class Acc: {0: 0.5, 1: 0.4, 2: 0.3, ...}
```

---

## 📦 Feature Extraction
- **Visual Features**: ResNet-101
- **Textual Features**: BLIP for caption generation + BERT for embedding
- These features are automatically integrated into graph node features when image paths are provided.

---

## 📝 Notes
- If you don't provide a dataset, the code runs with **dummy graphs** for testing the architecture.
- For real experiments (e.g., PISC dataset), replace dummy paths in `IPRDataset` with actual image paths and labels.

---

## 📄 Citation
If you use this code, please cite:
```
@inproceedings{akay2025dualgraph,
  title={Dual-Graph Transformer with Contextual Support Nodes for Interpersonal Relationship Recognition},
  author={Akay, Simge and Cakir, Duygu and Arica, Nafiz},
  booktitle={International Conference on Image Processing, Theory, Tools and Applications (IPTA)},
  year={2025},
  organization={IEEE}
}
```
