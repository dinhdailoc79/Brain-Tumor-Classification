# 🧠 TrustMed-AI

**Uncertainty-Aware Brain Tumor MRI Classification with Selective Prediction**

> *"A medical AI system that knows when it doesn't know is safer than one that is always confident."*

---

## Overview

TrustMed-AI is an end-to-end deep learning framework for **brain tumor classification** from MRI images. Beyond achieving high accuracy, the system quantifies prediction uncertainty and **automatically abstains** from unreliable diagnoses — a critical requirement for clinical safety.

**4 classes:** Glioma · Meningioma · Pituitary · No Tumor

---

## Key Features

| Feature | Description |
|---------|-------------|
| **NeuroFusionNet** | Dual-backbone architecture fusing Swin-V2 (global context) + ConvNeXt (local texture) via CLS-level attention |
| **Multi-Task Heads** | 3 expert heads mimicking clinical reasoning: tumor detection → classification → fine-grained discrimination |
| **Uncertainty Engine** | Monte Carlo Dropout (T=30) + post-hoc Temperature Scaling for calibrated confidence |
| **Selective Prediction** | 3-tier triage: **AUTO** (autonomous) · **ASSIST** (specialist review) · **ABSTAIN** (rejected) |
| **Explainability** | Grad-CAM++, Multi-Layer CAM, HiResCAM with GrabCut contour refinement |
| **Robustness Testing** | Corruption simulation (noise, blur, contrast shift) for domain shift evaluation |
| **Interactive Demo** | Gradio web app generating structured clinical reports |

---

## Project Structure

```
TrustMed-AI/
├── final_project/
│   ├── cell_00_load_checkpoints.py   # Load trained model checkpoints
│   ├── cell_01_setup.py              # Config, dependencies, GPU setup
│   ├── cell_02_data.py               # Dataset loading, augmentation, CLAHE
│   ├── cell_03_models.py             # Baselines + NeuroFusionNet architecture
│   ├── cell_04_training.py           # 2-phase training loop (freeze → fine-tune)
│   ├── cell_05_train_all.py          # Train all models sequentially
│   ├── cell_06_uncertainty.py        # MC Dropout, calibration, selective prediction
│   ├── cell_07_evaluation.py         # Metrics, confusion matrix, statistical tests
│   ├── cell_08_explainability.py     # Grad-CAM++, attention visualization
│   ├── cell_09_robustness.py         # Corruption & ablation analysis
│   ├── cell_10_demo.py               # Gradio interactive demo
│   ├── cell_11_radiomics.py          # Attention-driven morphological descriptors
│   ├── paper_draft.md                # Research paper draft
│   └── brain_tumor_dataset/          # MRI images (auto-downloaded)
├── checkpoints/                      # Saved model weights
└── results/                          # Output plots & metrics
```

---

## Requirements

- Python ≥ 3.10
- NVIDIA GPU recommended (A100/V100 for optimal performance; CPU supported but slow)

### Dependencies

```
torch>=2.0        torchvision>=0.15     timm>=0.9
albumentations>=1.3   grad-cam>=1.4     gradio>=4.0
numpy>=1.24       pandas>=2.0           matplotlib>=3.7
seaborn>=0.12     scikit-learn>=1.2     Pillow>=9.0
tqdm>=4.65        opencv-python-headless>=4.7
```

Missing packages are **auto-installed** at runtime.

---

## Quick Start

### 1. Clone & Setup

```bash
cd TrustMed-AI/final_project
pip install -r ../../requirements.txt
```

### 2. Train All Models

```bash
python cell_05_train_all.py
```

This trains 4 baselines (EfficientNetV2-S, ConvNeXt-Base, ViT-B/16, Swin-V2-S) + NeuroFusionNet with 2-phase training. The dataset is downloaded automatically via Kaggle API.

### 3. Run Uncertainty Analysis

```bash
python cell_06_uncertainty.py
```

Runs MC Dropout inference, temperature scaling calibration, and selective prediction analysis.

### 4. Run Evaluation & Explainability

```bash
python cell_07_evaluation.py    # Metrics & statistical tests
python cell_08_explainability.py # Grad-CAM++ visualizations
python cell_09_robustness.py    # Corruption robustness tests
python cell_11_radiomics.py     # Morphological descriptors
```

### 5. Launch Interactive Demo

```bash
python cell_10_demo.py
```

Opens a Gradio web interface for single-image inference with uncertainty visualization and clinical report generation.

---

## Architecture

```
                ┌──────────────┐     ┌──────────────┐
   MRI Input →  │  Swin-V2-S   │     │ ConvNeXt-Base │
                │  (global)    │     │  (local)     │
                └──────┬───────┘     └──────┬───────┘
                       │     CLS tokens     │
                       └────────┬───────────┘
                                │
                     ┌──────────▼──────────┐
                     │ Cross-Attention      │
                     │ Fusion (512-dim)     │
                     └──────────┬──────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
        ┌───────────┐   ┌───────────┐   ┌───────────┐
        │ Expert 1  │   │ Expert 2  │   │ Expert 3  │
        │ Tumor?    │   │ 4-class   │   │ Glioma vs │
        │ (2-cls)   │   │ classify  │   │ Mening.   │
        └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
              └────────────────┼────────────────┘
                               ▼
                     Gated Expert Voting
                               ▼
                    Final 4-class Prediction
                               ▼
                  ┌────────────────────────┐
                  │   Uncertainty Engine   │
                  │  MC Dropout + Temp.    │
                  │     Scaling            │
                  └────────────┬───────────┘
                               ▼
                  AUTO · ASSIST · ABSTAIN
```

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Image size | 224×224 (256×256 for Swin-V2) |
| Phase 1 | 5 epochs, backbone frozen, lr=1e-3 |
| Phase 2 | 25 epochs, full fine-tune, backbone lr=2e-5, head lr=5e-4 |
| Optimizer | AdamW (weight decay=0.01) |
| Regularization | Label smoothing=0.1, MixUp α=0.2, CutMix α=1.0 |
| MC Dropout | T=30 forward passes |
| Seed | 42 |

---

## Dataset

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) — ~7,000 2D MRI slices across 4 classes.

Auto-downloaded via Kaggle API. Place your `kaggle.json` in `~/.kaggle/` or set `KAGGLE_USERNAME` & `KAGGLE_KEY` environment variables.

---

## License

This project is for academic research purposes (FPT University — DPL302m).
