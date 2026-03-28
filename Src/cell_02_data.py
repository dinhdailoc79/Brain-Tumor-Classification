import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from cell_01_setup import *

# %% Cell 2: Dataset Loading & Exploratory Data Analysis
# ════════════════════════════════════════════════════════════════════════
# INSIGHT (Clinical AI): Thorough EDA reveals the data biases that cause
# false-positive rates of 35% (YOLOv7, 2024). We must understand our
# dataset's structure BEFORE designing the model.
# ════════════════════════════════════════════════════════════════════════

def find_data_root(base_dir: str) -> str:
    """Auto-detect data root containing Training/ and Testing/."""
    if os.path.isdir(os.path.join(base_dir, 'Training')):
        return base_dir
    for sub in sorted(os.listdir(base_dir)):
        sub_path = os.path.join(base_dir, sub)
        if os.path.isdir(sub_path):
            if os.path.isdir(os.path.join(sub_path, 'Training')):
                return sub_path
            for sub2 in sorted(os.listdir(sub_path)):
                sub2_path = os.path.join(sub_path, sub2)
                if os.path.isdir(sub2_path) and os.path.isdir(os.path.join(sub2_path, 'Training')):
                    return sub2_path
    return base_dir


def load_dataset(data_dir: str) -> pd.DataFrame:
    """Build DataFrame of all images with labels and splits."""
    root = find_data_root(data_dir)
    if root != data_dir:
        print(f"📂 Auto-detected data root: {root}")

    records = []
    for split in ['Training', 'Testing']:
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            split_dir = os.path.join(root, split.lower())
        if not os.path.isdir(split_dir):
            print(f"⚠️ Missing: {split_dir}")
            continue
        for cls_name in sorted(os.listdir(split_dir)):
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif')):
                    records.append({
                        'path': os.path.join(cls_dir, fname),
                        'label': cls_name.lower().replace(' ', '').replace('_', ''),
                        'split': split,
                    })

    df = pd.DataFrame(records)
    if len(df) == 0:
        raise FileNotFoundError(f"❌ No images found in {root}. Check dataset structure.")

    print(f"✅ Loaded {len(df)} images from {root}")
    return df


# ════════════════════════════════════════════════════════════════════════
# DATASET CLASS & TRANSFORMS
# ════════════════════════════════════════════════════════════════════════

class BrainTumorDataset(Dataset):
    """Brain Tumor MRI Dataset with Albumentations transforms."""

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = np.array(Image.open(row['path']).convert('RGB'))

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        label = int(row['label_idx'])
        return img, label


def get_transforms(mode: str, img_size: int = 224):
    """Create augmentation pipelines for train/val/test/TTA."""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if mode == 'train':
        # ═══════════════════════════════════════════════════════════════
        # Q1 MEDICAL AUGMENTATION POLICY:
        # ───────────────────────────────────────────────────────────────
        # ✅ HorizontalFlip: Brain has approximate bilateral symmetry
        # ❌ VerticalFlip: REMOVED — brain anatomy is NOT top-bottom
        #    symmetric. Flipping upside-down creates anatomically
        #    impossible images → model learns artifacts, not pathology
        # ✅ Mild rotation (±15°): Simulates slight head tilt in scanner
        # ✅ CLAHE: Enhances local contrast (standard in medical imaging)
        # ✅ Elastic distortion: Simulates tissue deformation
        # ✅ Gaussian noise: Simulates scanner noise
        # ⚠️ Brightness/Contrast: CONSERVATIVE limits — MRI intensity
        #    carries diagnostic meaning (e.g., T1 enhancement patterns)
        # ❌ Color jittering: EXCLUDED — MRI is grayscale, color shifts
        #    destroy pulse sequence semantics
        # ═══════════════════════════════════════════════════════════════
        return A.Compose([
            A.Resize(img_size, img_size),
            # ── Medical-specific preprocessing ──
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            # ── Spatial augmentations (anatomically valid) ──
            A.HorizontalFlip(p=0.5),  # Brain bilateral symmetry OK
            # NO VerticalFlip — anatomically invalid for brain MRI
            A.Rotate(limit=15, border_mode=0, p=0.5),  # Mild head tilt
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                               rotate_limit=10, border_mode=0, p=0.3),
            # ── Elastic/Grid distortion (simulate tissue deformation) ──
            A.OneOf([
                A.ElasticTransform(alpha=60, sigma=60*0.05, p=0.3),
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=0.3),
            ], p=0.2),
            # ── Intensity augmentations (CONSERVATIVE for MRI) ──
            A.RandomBrightnessContrast(brightness_limit=0.1,
                                       contrast_limit=0.1, p=0.2),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.15),
            A.GaussianBlur(blur_limit=(3, 5), p=0.1),
            # ── Regularization ──
            A.CoarseDropout(max_holes=4, max_height=img_size//10,
                            max_width=img_size//10, fill_value=0, p=0.2),
            # ── Normalize + ToTensor ──
            A.Normalize(mean=imagenet_mean, std=imagenet_std),
            ToTensorV2(),
        ])

    elif mode in ['val', 'test']:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=imagenet_mean, std=imagenet_std),
            ToTensorV2(),
        ])

    elif mode == 'tta':
        # TTA: Only anatomically valid augmentations
        base = [A.Resize(img_size, img_size)]
        norm = [A.Normalize(mean=imagenet_mean, std=imagenet_std), ToTensorV2()]
        return [
            A.Compose(base + norm),  # Original
            A.Compose(base + [A.HorizontalFlip(p=1.0)] + norm),  # Bilateral symmetry
            # NO VerticalFlip — anatomically invalid
            A.Compose(base + [A.Rotate(limit=(5, 5), p=1.0)] + norm),  # Mild tilt
            A.Compose(base + [A.Rotate(limit=(-5, -5), p=1.0)] + norm),  # Opposite tilt
            A.Compose(base + [A.CLAHE(clip_limit=2.0, p=1.0)] + norm),  # Contrast enhance
        ]


def get_model_img_size(model_key: str) -> int:
    """Return required image size for a given model."""
    if 'swin' in model_key.lower() and '256' in model_key:
        return 256
    return 224


def get_balanced_sampler(labels: List[int]) -> WeightedRandomSampler:
    """Create weighted sampler for class-balanced training."""
    counts = Counter(labels)
    weight_per_class = {cls: 1.0 / cnt for cls, cnt in counts.items()}
    weights = [weight_per_class[label] for label in labels]
    return WeightedRandomSampler(weights, len(weights), replacement=True)


def setup_data():
    """Initialize all data: load dataset, EDA, splits, and DataLoaders.

    Must be called explicitly (not at import time) to avoid Windows
    multiprocessing spawn issues.

    Returns a dict with all data objects for downstream cells.
    """
    global df, label2idx, idx2label
    global train_df, test_df, val_df, train_df_final, kfold_splits
    global train_ds, val_ds, test_ds
    global train_loader, val_loader, test_loader

    # ── Load dataset ──
    df = load_dataset(cfg.data_dir)

    # ── Map labels ──
    label2idx = {name: idx for idx, name in enumerate(cfg.class_names)}
    idx2label = {v: k for k, v in label2idx.items()}
    df['label_idx'] = df['label'].map(label2idx)

    # Verify all labels mapped correctly
    unmapped = df[df['label_idx'].isna()]
    if len(unmapped) > 0:
        print(f"⚠️ Unmapped labels: {unmapped['label'].unique()}")
        label_fixes = {
            'notumor': 'notumor', 'no_tumor': 'notumor', 'no tumor': 'notumor',
            'healthy': 'notumor', 'normal': 'notumor',
        }
        df['label'] = df['label'].map(lambda x: label_fixes.get(x, x))
        df['label_idx'] = df['label'].map(label2idx)
        df = df.dropna(subset=['label_idx'])
        df['label_idx'] = df['label_idx'].astype(int)

    print(f"\n📊 Label distribution:")
    for split in ['Training', 'Testing']:
        sub = df[df['split'] == split]
        counts = sub['label'].value_counts()
        print(f"   {split}: {dict(counts)} (total: {len(sub)})")

    # ════════════════════════════════════════════════════════════════════════
    # EDA PLOTS (Publication Quality)
    # ════════════════════════════════════════════════════════════════════════

    # ── 1. Class Distribution ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, split in enumerate(['Training', 'Testing']):
        sub = df[df['split'] == split]
        counts = sub['label'].value_counts().reindex(cfg.class_names)
        bars = axes[i].bar(cfg.class_names, counts,
                           color=[CLASS_COLORS[c] for c in cfg.class_names],
                           edgecolor='black', linewidth=0.5)
        axes[i].set_title(f'{split} Set Distribution', fontsize=14, fontweight='bold')
        axes[i].set_ylabel('Number of Images')
        for bar, cnt in zip(bars, counts):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                         str(int(cnt)), ha='center', va='bottom', fontweight='bold')
        ratio = counts.max() / counts.min()
        axes[i].text(0.95, 0.95, f'Imbalance ratio: {ratio:.2f}',
                     transform=axes[i].transAxes, ha='right', va='top',
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, 'fig_class_distribution.png'))
    plt.show()

    # ── 2. Sample MRI images per class ──
    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    for row, cls_name in enumerate(cfg.class_names):
        cls_imgs = df[(df['label'] == cls_name) & (df['split'] == 'Training')].sample(
            6, random_state=cfg.seed
        )
        for col, (_, r) in enumerate(cls_imgs.iterrows()):
            img = Image.open(r['path']).convert('RGB')
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_ylabel(cls_name.upper(), fontsize=12,
                                           fontweight='bold', rotation=90, labelpad=15)
    fig.suptitle('Brain Tumor MRI — Sample Images per Class', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, 'fig_sample_images.png'))
    plt.show()

    # ── 3. Image size statistics ──
    print("\n📏 Image Size Statistics (sample of 200):")
    sample_paths = df.sample(min(200, len(df)), random_state=cfg.seed)['path']
    sizes = [Image.open(p).size for p in sample_paths]
    widths, heights = zip(*sizes)
    print(f"   Width:  min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.0f}")
    print(f"   Height: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.0f}")

    # ════════════════════════════════════════════════════════════════════════
    # STRATIFIED SPLITTING: Train / Val / Test
    # ════════════════════════════════════════════════════════════════════════
    # Q1 RIGOR — Data Leakage Analysis:
    # ─────────────────────────────────
    # The masoudnickparvar/brain-tumor-mri-dataset provides pre-cropped 2D
    # slices without patient identifiers. This means:
    #   1. We CANNOT perform patient-level splitting (no patient IDs available)
    #   2. There is a RISK that multiple slices from the same patient exist
    #      in both Train and Test sets (potential data leakage)
    #   3. The original dataset provides Train/Test split — we RESPECT this
    #      split rather than re-shuffling (preserves original authors' intent)
    #   4. We ONLY split within Training → Train + Val (stratified)
    #
    # LIMITATION ACKNOWLEDGED IN PAPER:
    #   "As the dataset lacks patient-level identifiers, we cannot guarantee
    #    patient-level isolation between splits. This is a known limitation
    #    shared by all studies using this benchmark dataset (VGG16 2025,
    #    Swin+AE-cGAN 2025, etc.)."
    #
    # NORMALIZATION:
    #   We use ImageNet mean/std (standard for transfer learning from
    #   pretrained models). These statistics are computed from ImageNet,
    #   NOT from our dataset → no data leakage from normalization.
    # ════════════════════════════════════════════════════════════════════════

    train_df = df[df['split'] == 'Training'].reset_index(drop=True)
    test_df  = df[df['split'] == 'Testing'].reset_index(drop=True)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=cfg.seed)
    train_idx, val_idx = next(sss.split(train_df['path'], train_df['label_idx']))

    val_df = train_df.iloc[val_idx].reset_index(drop=True)
    train_df_final = train_df.iloc[train_idx].reset_index(drop=True)

    print(f"\n📊 Final Splits (Stratified):")
    print(f"   Train: {len(train_df_final)} images")
    print(f"   Val:   {len(val_df)} images")
    print(f"   Test:  {len(test_df)} images (original held-out)")
    print(f"   ⚠️  Note: No patient IDs → patient-level split not possible")

    for name, sub_df in [('Train', train_df_final), ('Val', val_df), ('Test', test_df)]:
        counts = sub_df['label'].value_counts()
        print(f"   {name}: {dict(counts)}")

    # ── Stratified K-Fold indices (for robustness validation) ──
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed)
    kfold_splits = list(kfold.split(train_df['path'], train_df['label_idx']))
    print(f"   📁 5-Fold CV indices precomputed (available for ablation)")

    # ── Create standard dataloaders ──
    train_ds = BrainTumorDataset(train_df_final, get_transforms('train', cfg.img_size))
    val_ds   = BrainTumorDataset(val_df, get_transforms('val', cfg.img_size))
    test_ds  = BrainTumorDataset(test_df, get_transforms('test', cfg.img_size))

    train_sampler = get_balanced_sampler(train_df_final['label_idx'].tolist())

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=train_sampler,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    # Quick sanity check
    batch_imgs, batch_labels = next(iter(train_loader))
    print(f"\n✅ DataLoader OK: batch shape = {batch_imgs.shape}, labels = {batch_labels[:8].tolist()}")
    print(f"   Pixel range: [{batch_imgs.min():.3f}, {batch_imgs.max():.3f}]")

    return {
        'df': df, 'label2idx': label2idx, 'idx2label': idx2label,
        'train_df_final': train_df_final, 'val_df': val_df, 'test_df': test_df,
        'train_ds': train_ds, 'val_ds': val_ds, 'test_ds': test_ds,
        'train_loader': train_loader, 'val_loader': val_loader, 'test_loader': test_loader,
        'kfold_splits': kfold_splits,
    }


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    setup_data()
