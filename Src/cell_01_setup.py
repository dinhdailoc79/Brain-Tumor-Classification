# ════════════════════════════════════════════════════════════════════════
# 🧠 TrustMed-AI: Trustworthy Brain Tumor MRI Classification
# ════════════════════════════════════════════════════════════════════════
# Uncertainty-Aware Hierarchical Multi-Task Expert Voting with Selective Prediction
#
# Core Thesis: "A medical AI system that knows when it doesn't know
#               is safer than one that is always confident."
#
# Pipeline Overview:
#   1. Data Loading & Medical-Specific EDA
#   2. CLAHE Preprocessing & Domain-Robust Augmentation
#   3. 4 SOTA Baselines (CNN → Transformer spectrum)
#   4. NeuroFusionNet: Dual-Backbone + CLS-Level Attention Fusion + Multi-Task Heads
#   5. ★ Uncertainty Engine (MC Dropout, Post-Hoc Temperature Scaling)
#   6. ★ Selective Prediction (AUTO / ASSIST / ABSTAIN)
#   7. Explainability (Grad-CAM++, Multi-Layer CAM, HiResCAM, Expert Routing)
#   8. Robustness Evaluation (Corruption, Ablation, Statistical Tests)
#   9. Morphological Descriptors (Attention-Driven Pseudo-Radiomics)
#  10. Interactive Demo (Gradio)
# ════════════════════════════════════════════════════════════════════════

# %% Cell 1: Install Dependencies & Imports
import subprocess, sys

# pip-name → import-name (for packages where they differ)
_REQUIRED = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'Pillow': 'PIL',
    'scikit-learn': 'sklearn',
    'tqdm': 'tqdm',
    'torch': 'torch',
    'torchvision': 'torchvision',
    'timm': 'timm',
    'albumentations': 'albumentations',
    'grad-cam': 'grad_cam',
    'gradio': 'gradio',
    'scikit-image': 'skimage',
}
for pip_name, import_name in _REQUIRED.items():
    try:
        __import__(import_name)
    except ImportError:
        print(f"Đang cài đặt thư viện bị thiếu: {pip_name} (Việc này có thể mất vài phút)...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pip_name])

# ── Standard Library ──
import os, random, copy, warnings, time, gc, json, math
from pathlib import Path
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union

# ── Scientific Computing ──
import numpy as np
import pandas as pd

# ── Visualization ──
import matplotlib
matplotlib.use('Agg')  # Non-blocking backend for .py scripts (plots saved to files)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image

# ── PyTorch Core ──
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel

# ── Pretrained Models ──
import timm

# ── Sklearn Metrics ──
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve, auc, cohen_kappa_score, matthews_corrcoef,
    precision_recall_curve, average_precision_score, brier_score_loss,
    log_loss
)
from sklearn.manifold import TSNE
from sklearn.calibration import calibration_curve

# ── Image Augmentation ──
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Progress Bars ──
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════════════
# PUBLICATION-QUALITY PLOT SETTINGS
# ════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'figure.figsize': (10, 8),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})
COLORS = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#F4A261']
CLASS_COLORS = {
    'glioma': '#E63946',
    'meningioma': '#457B9D',
    'notumor': '#2A9D8F',
    'pituitary': '#E9C46A',
}

# ════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    """Centralized configuration for TrustMed-AI."""

    # ── Paths (relative to project root) ──
    project_root: str = os.path.dirname(os.path.abspath(__file__))
    data_dir: str = ''
    output_dir: str = ''
    checkpoint_dir: str = ''

    def __post_init__(self):
        self.data_dir = self.data_dir or os.path.join(self.project_root, 'brain_tumor_dataset')
        self.output_dir = self.output_dir or os.path.join(self.project_root, 'trustmed_results')
        self.checkpoint_dir = self.checkpoint_dir or os.path.join(self.project_root, 'trustmed_checkpoints')

    # ── Classes ──
    class_names: List[str] = field(default_factory=lambda: [
        'glioma', 'meningioma', 'notumor', 'pituitary'
    ])
    num_classes: int = 4

    # ── Image ──
    img_size: int = 224
    img_size_swin: int = 256  # Swin-V2 needs 256

    # ── Training — A100 Optimized ──
    batch_size: int = 32
    num_workers: int = 0 if sys.platform == 'win32' else 2
    persistent_workers: bool = True

    # Phase 1: Frozen backbone (warm up heads)
    phase1_epochs: int = 5
    phase1_lr: float = 1e-3

    # Phase 2: Full fine-tune
    phase2_epochs: int = 25
    phase2_backbone_lr: float = 2e-5
    phase2_head_lr: float = 5e-4

    # ── Regularization ──
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    mix_prob: float = 0.5
    dropout: float = 0.3

    # ── Early stopping ──
    patience: int = 8
    gradient_clip: float = 1.0

    # ── AMP ──
    amp_dtype: str = 'bfloat16'

    # ── EMA ──
    ema_decay: float = 0.999

    # ── Uncertainty ★ ──
    mc_dropout_T: int = 30       # MC Dropout forward passes
    ensemble_N: int = 3          # Deep Ensemble models (lighter for A100 time)
    temperature_init: float = 1.5

    # ── Selective Prediction ★ ──
    theta_high: float = 0.85     # AUTO threshold (will be optimized)
    theta_low: float = 0.50      # ABSTAIN threshold (will be optimized)

    # ── Reproducibility ──
    seed: int = 42

    # ── Models ──
    baseline_models: Dict[str, str] = field(default_factory=lambda: OrderedDict({
        'EfficientNetV2-S': 'tf_efficientnetv2_s',
        'ConvNeXt-Base':    'convnext_base',
        'ViT-B/16':        'vit_base_patch16_224',
        'Swin-V2-S':       'swinv2_small_window8_256',
    }))

    # ── Loss weights (multi-task) ──
    w_main: float = 1.0
    w_tumor: float = 0.3
    w_fine: float = 0.5      # High: glioma↔meningioma is the bottleneck
    w_contrastive: float = 0.2

    # ── NeuroFusionNet ──
    fusion_dim: int = 512
    num_attention_heads: int = 8


cfg = Config()
os.makedirs(cfg.output_dir, exist_ok=True)
os.makedirs(cfg.checkpoint_dir, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════
# REPRODUCIBILITY
# ════════════════════════════════════════════════════════════════════════
def seed_everything(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # False for speed on A100
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(cfg.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ════════════════════════════════════════════════════════════════════════
# A100 GPU MAXIMUM UTILIZATION
# ════════════════════════════════════════════════════════════════════════
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    gpu_cap = torch.cuda.get_device_capability(0)
    print(f"🚀 GPU: {gpu_name} | {gpu_mem:.1f} GB | SM {gpu_cap[0]}.{gpu_cap[1]}")

    # TF32: 8x faster matmul on A100
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    # BF16 or FP16
    if gpu_cap[0] >= 8:  # Ampere+
        cfg.amp_dtype = 'bfloat16'
        print("   ✅ BF16 autocast (native A100)")
    else:
        cfg.amp_dtype = 'float16'
        print("   ⚠️ FP16 autocast (pre-Ampere)")

    # Flash Attention
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        print("   ✅ Flash Attention (SDPA)")

    # Batch size auto-adjust
    if gpu_mem >= 75:
        cfg.batch_size = 96
    elif gpu_mem >= 35:
        cfg.batch_size = 64
    elif gpu_mem >= 6:
        cfg.batch_size = 32
    else:
        cfg.batch_size = 16

    print(f"   ✅ Batch size: {cfg.batch_size} | Workers: {cfg.num_workers}")
    print(f"   ✅ TF32 + cuDNN benchmark enabled")
else:
    print("⚠️ No GPU detected — running on CPU (training will be slow).")
    cfg.amp_dtype = 'float32'
    cfg.batch_size = 8
    cfg.num_workers = 0
    cfg.persistent_workers = False

# ════════════════════════════════════════════════════════════════════════
# DATASET DOWNLOAD (Kaggle API — cross-platform)
# ════════════════════════════════════════════════════════════════════════
import zipfile, shutil

def _download_dataset(data_dir: str):
    """Download Brain Tumor MRI Dataset via Kaggle API (cross-platform)."""
    print("\n📥 Downloading Brain Tumor MRI Dataset...")

    # Ensure KAGGLE_USERNAME / KAGGLE_KEY are set in env vars or ~/.kaggle/kaggle.json
    if not (os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY')):
        kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
        if not kaggle_json.exists():
            print("   ⚠️  Set KAGGLE_USERNAME & KAGGLE_KEY env vars,")
            print("       or place kaggle.json in ~/.kaggle/")
            return

    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'kaggle'])
    subprocess.check_call([
        sys.executable, '-m', 'kaggle', 'datasets', 'download',
        '-d', 'masoudnickparvar/brain-tumor-mri-dataset',
        '-p', os.path.dirname(data_dir),
    ])

    zip_path = os.path.join(os.path.dirname(data_dir), 'brain-tumor-mri-dataset.zip')
    if os.path.isfile(zip_path):
        os.makedirs(data_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(data_dir)
        os.remove(zip_path)
        print(f"✅ Dataset ready at: {data_dir}")
    else:
        print("   ❌ Download failed — zip file not found.")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    
    if not os.path.isdir(cfg.data_dir) or len(os.listdir(cfg.data_dir)) == 0:
        _download_dataset(cfg.data_dir)
    else:
        print(f"\n✅ Dataset exists: {cfg.data_dir}")

    print(f"\n{'='*60}")
    print(f"🧠 TrustMed-AI: Trustworthy Brain Tumor Classification")
    print(f"{'='*60}")
    print(f"📂 Data: {cfg.data_dir}")
    print(f"📊 Classes: {cfg.class_names}")
    print(f"🎯 Device: {device} | AMP: {cfg.amp_dtype}")
    print(f"🔬 Uncertainty: MC-T={cfg.mc_dropout_T} | Ensemble-N={cfg.ensemble_N}")
    print(f"{'='*60}")
