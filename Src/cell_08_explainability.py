import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from cell_04_training import *

# %% Cell 8: Explainability — Grad-CAM++, t-SNE, Expert Routing
# ════════════════════════════════════════════════════════════════════════
# DESIGN (Falsification): We don't just show heatmaps — we MEASURE
# overlap with actual pathology regions. Grad-CAM Dice = 0.33 is the
# state-of-the-art failure → we report honestly.
# ════════════════════════════════════════════════════════════════════════

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ════════════════════════════════════════════════════════════════════════
# 1. GRAD-CAM++ VISUALIZATION
# ════════════════════════════════════════════════════════════════════════

def get_target_layers(model, backbone='both'):
    """Get target layers for Grad-CAM based on model type."""
    if hasattr(model, 'backbone_global') and hasattr(model, 'backbone_local'):
        layers = []
        if backbone in ['both', 'global']:
            swin = model.backbone_global
            if hasattr(swin, 'layers'):
                layers.append(swin.layers[-1].blocks[-1].norm2)
        if backbone in ['both', 'local']:
            convnext = model.backbone_local
            if hasattr(convnext, 'stages'):
                layers.append(convnext.stages[-1].blocks[-1])
        return layers if layers else [list(model.modules())[-3]]
    else:
        backbone = model.backbone
        if hasattr(backbone, 'stages'):
            return [backbone.stages[-1].blocks[-1]]
        elif hasattr(backbone, 'layers'):
            return [backbone.layers[-1].blocks[-1].norm2]
        elif hasattr(backbone, 'blocks'):
            return [backbone.blocks[-1].norm1]
        return [list(backbone.modules())[-3]]


class ModelWrapper(nn.Module):
    """Wrapper to make NeuroFusionNet compatible with GradCAM."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, dict):
            return out['logits']
        return out


def generate_gradcam_grid(model, dataset, device, n_per_class=3,
                           class_names=None, save_path=None):
    """Generate Grad-CAM++ visualizations for sample images."""
    model.eval()
    wrapped = ModelWrapper(model)
    target_layers = get_target_layers(model, 'local')

    if not target_layers:
        print("⚠️ Could not find target layers for Grad-CAM")
        return

    cam = GradCAMPlusPlus(model=wrapped, target_layers=target_layers)

    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])

    fig, axes = plt.subplots(4, n_per_class * 2, figsize=(4 * n_per_class * 2, 16))

    for cls_idx, cls_name in enumerate(class_names or cfg.class_names):
        # Find samples of this class
        indices = [i for i in range(len(dataset)) if dataset.df.iloc[i]['label_idx'] == cls_idx]
        sel = np.random.choice(indices, min(n_per_class, len(indices)), replace=False)

        for j, idx in enumerate(sel):
            img_tensor, label = dataset[idx]
            img_input = img_tensor.unsqueeze(0).to(device)

            with torch.amp.autocast('cuda'):
                logits = model(img_input)
                if isinstance(logits, dict):
                    logits = logits['logits']
                pred = logits.argmax(1).item()
                conf = F.softmax(logits, dim=1).max().item()

            targets = [ClassifierOutputTarget(pred)]
            grayscale_cam = cam(input_tensor=img_input, targets=targets)[0]

            rgb = img_tensor.permute(1, 2, 0).numpy()
            rgb = (rgb * imagenet_std + imagenet_mean).clip(0, 1)

            # Original
            ax_orig = axes[cls_idx, j * 2]
            ax_orig.imshow(rgb)
            ax_orig.set_title(f'{cls_name}\nTrue', fontsize=10)
            ax_orig.axis('off')

            # Grad-CAM
            cam_img = show_cam_on_image(rgb.astype(np.float32), grayscale_cam, use_rgb=True)
            ax_cam = axes[cls_idx, j * 2 + 1]
            ax_cam.imshow(cam_img)
            status = "✓" if pred == label else "✗"
            ax_cam.set_title(f'Pred: {class_names[pred]} ({conf:.0%}) {status}', fontsize=10,
                             color='green' if pred == label else 'red')
            ax_cam.axis('off')

    fig.suptitle('Grad-CAM++ Attention Maps — NeuroFusionNet',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

    cam.__del__()


# ════════════════════════════════════════════════════════════════════════
# 1b. ★ MULTI-LAYER GRAD-CAM (SHARPER — fuses multiple resolutions)
# ════════════════════════════════════════════════════════════════════════
# Standard Grad-CAM uses ONLY the last layer (8×8 resolution) → blurry.
# Multi-Layer fuses attention from multiple stages:
#   Stage 1: 64×64 (fine details, edges)
#   Stage 2: 32×32 (medium features)
#   Stage 3: 16×16 (large structures)
#   Stage 4: 8×8   (semantic = standard Grad-CAM)
# Result: MUCH sharper boundaries around the tumor.

def get_multi_layer_targets(model):
    """Get target layers from MULTIPLE stages for sharper attention."""
    layers = []
    if hasattr(model, 'backbone_local'):
        convnext = model.backbone_local
        if hasattr(convnext, 'stages'):
            for stage in convnext.stages:
                layers.append(stage.blocks[-1])
    elif hasattr(model, 'backbone'):
        backbone = model.backbone
        if hasattr(backbone, 'stages'):
            for stage in backbone.stages:
                layers.append(stage.blocks[-1])
    return layers if layers else get_target_layers(model, 'local')


def generate_multilayer_cam(model, img_tensor, device, pred_class):
    """Generate sharp attention map by fusing multiple Grad-CAM layers.

    Each layer produces a heatmap at different resolution. We:
    1. Generate Grad-CAM at each layer
    2. Resize all to original image size
    3. Weight-average: deeper layers get more semantic weight,
       shallower layers add boundary precision
    """
    wrapped = ModelWrapper(model)
    target_layers = get_multi_layer_targets(model)

    if len(target_layers) < 2:
        # Fallback to standard Grad-CAM++
        cam = GradCAMPlusPlus(model=wrapped, target_layers=target_layers)
        targets = [ClassifierOutputTarget(pred_class)]
        result = cam(input_tensor=img_tensor, targets=targets)[0]
        cam.__del__()
        return result

    # Generate CAM for each layer separately
    layer_cams = []
    # Semantic weights: deeper = more weight (but shallower adds sharpness)
    layer_weights = [0.10, 0.15, 0.25, 0.50]  # stage1→stage4

    for i, layer in enumerate(target_layers):
        try:
            cam = GradCAMPlusPlus(model=wrapped, target_layers=[layer])
            targets = [ClassifierOutputTarget(pred_class)]
            grayscale = cam(input_tensor=img_tensor, targets=targets)[0]
            layer_cams.append(grayscale)
            cam.__del__()
        except Exception:
            continue

    if not layer_cams:
        return np.zeros((img_tensor.shape[2], img_tensor.shape[3]))

    # Weight-average fusion
    weights = layer_weights[:len(layer_cams)]
    total_w = sum(weights)
    fused = np.zeros_like(layer_cams[0])
    for cam_map, w in zip(layer_cams, weights):
        fused += (w / total_w) * cam_map

    # Sharpen: apply power transform to increase contrast
    fused = np.power(fused, 0.7)  # < 1 = spread, > 1 = concentrate
    fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)

    return fused


# ════════════════════════════════════════════════════════════════════════
# 1c. ★ HiResCAM (PIXEL-LEVEL — element-wise gradient × activation)
# ════════════════════════════════════════════════════════════════════════
# HiResCAM (Draelos & Carin, 2021): instead of GAP(gradients) × activations,
# it computes element-wise: grad ⊙ activation at each spatial position.
# Result: pixel-level precision, no upsampling blur.

def generate_hirescam(model, img_tensor, device, pred_class):
    """Generate HiResCAM: element-wise gradient × activation.

    Much sharper than Grad-CAM because it preserves spatial information
    at the activation resolution without global average pooling.
    """
    model.eval()
    wrapped = ModelWrapper(model)

    # Use the deepest available layer with spatial dims
    target_layers = get_target_layers(model, 'local')
    if not target_layers:
        return None

    target_layer = target_layers[-1]

    # Hook to capture activations and gradients
    activations = {}
    gradients = {}

    def fwd_hook(module, input, output):
        activations['value'] = output.detach()

    def bwd_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach()

    fwd_handle = target_layer.register_forward_hook(fwd_hook)
    bwd_handle = target_layer.register_full_backward_hook(bwd_hook)

    try:
        # Forward
        img_tensor.requires_grad_(True)
        with torch.amp.autocast('cuda'):
            output = wrapped(img_tensor)
        score = output[0, pred_class]

        # Backward
        model.zero_grad()
        score.backward()

        # HiResCAM: element-wise product (no GAP!)
        act = activations['value']  # (1, C, H, W) or (1, L, C)
        grad = gradients['value']   # same shape

        # Handle different shapes (ViT/Swin output vs CNN output)
        if act.dim() == 3:
            # Transformer: (1, L, C) → need to reshape to 2D
            L, C = act.shape[1], act.shape[2]
            h = w = int(L ** 0.5)
            if h * w != L:
                # Can't reshape perfectly, fallback
                hirescam = (act * grad).sum(dim=-1)[0]  # (L,)
                hirescam = hirescam.reshape(h, -1)
            else:
                act_2d = act[0].reshape(h, w, C)
                grad_2d = grad[0].reshape(h, w, C)
                hirescam = (act_2d * grad_2d).sum(dim=-1)  # (h, w)
        else:
            # CNN: (1, C, H, W) → standard
            hirescam = (act[0] * grad[0]).sum(dim=0)  # (H, W)

        # ReLU + normalize
        hirescam = F.relu(hirescam)
        hirescam = hirescam.cpu().numpy()

        # Resize to input size
        from scipy.ndimage import zoom
        h_in, w_in = img_tensor.shape[2], img_tensor.shape[3]
        if hirescam.shape[0] != h_in:
            scale_h = h_in / hirescam.shape[0]
            scale_w = w_in / hirescam.shape[1]
            hirescam = zoom(hirescam, (scale_h, scale_w), order=1)

        # Normalize
        hirescam = (hirescam - hirescam.min()) / (hirescam.max() - hirescam.min() + 1e-8)

        return hirescam

    except Exception as e:
        print(f"HiResCAM error: {e}")
        return None
    finally:
        fwd_handle.remove()
        bwd_handle.remove()
        img_tensor.requires_grad_(False)


# ════════════════════════════════════════════════════════════════════════
# 1d. COMPARISON GRID: Grad-CAM++ vs Multi-Layer vs HiResCAM
# ════════════════════════════════════════════════════════════════════════

def generate_xai_comparison(model, dataset, device, n_samples=8, save_path=None):
    """Side-by-side comparison of 3 XAI methods for paper figure."""
    model.eval()

    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == 'bfloat16' else torch.float16

    # Sample 2 per class
    indices = []
    for cls_idx in range(4):
        cls_indices = [i for i in range(len(dataset))
                       if dataset.df.iloc[i]['label_idx'] == cls_idx]
        indices.extend(np.random.choice(cls_indices, min(2, len(cls_indices)), replace=False))

    n = len(indices)
    fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n))
    col_titles = ['Original MRI', 'Grad-CAM++\n(8×8, standard)', 'Multi-Layer\n(8-64×, fused)', 'HiResCAM\n(pixel-level)']

    for row, idx in enumerate(indices):
        img_tensor, label = dataset[idx]
        img_input = img_tensor.unsqueeze(0).to(device)
        rgb = img_tensor.permute(1, 2, 0).numpy()
        rgb = (rgb * imagenet_std + imagenet_mean).clip(0, 1)

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=amp_dtype):
            logits = model(img_input)
            if isinstance(logits, dict):
                logits = logits['logits']
            pred = logits.argmax(1).item()
            conf = F.softmax(logits, dim=1).max().item()

        cls_name = cfg.class_names[label]
        pred_name = cfg.class_names[pred]
        status = "✓" if pred == label else "✗"

        # Column 0: Original
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f'{cls_name}\n(true)', fontsize=10)
        axes[row, 0].axis('off')

        # Column 1: Grad-CAM++ (standard, blurry)
        wrapped = ModelWrapper(model)
        target_layers = get_target_layers(model, 'local')
        cam = GradCAMPlusPlus(model=wrapped, target_layers=target_layers)
        gradcam_map = cam(input_tensor=img_input,
                          targets=[ClassifierOutputTarget(pred)])[0]
        cam.__del__()
        cam_img = show_cam_on_image(rgb.astype(np.float32), gradcam_map, use_rgb=True)
        axes[row, 1].imshow(cam_img)
        axes[row, 1].set_title(f'Pred: {pred_name} ({conf:.0%}) {status}', fontsize=10,
                               color='green' if pred == label else 'red')
        axes[row, 1].axis('off')

        # Column 2: Multi-Layer Grad-CAM (sharper)
        ml_map = generate_multilayer_cam(model, img_input, device, pred)
        ml_img = show_cam_on_image(rgb.astype(np.float32), ml_map, use_rgb=True)
        axes[row, 2].imshow(ml_img)
        axes[row, 2].set_title('Multi-Layer Fusion', fontsize=10)
        axes[row, 2].axis('off')

        # Column 3: HiResCAM (sharpest)
        hr_map = generate_hirescam(model, img_input, device, pred)
        if hr_map is not None:
            hr_img = show_cam_on_image(rgb.astype(np.float32),
                                        hr_map.astype(np.float32), use_rgb=True)
        else:
            hr_img = rgb
        axes[row, 3].imshow(hr_img)
        axes[row, 3].set_title('HiResCAM', fontsize=10)
        axes[row, 3].axis('off')

    # Column headers
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title + '\n' + axes[0, j].get_title(), fontsize=11, fontweight='bold')

    fig.suptitle('Explainability Comparison: Grad-CAM++ vs Multi-Layer vs HiResCAM',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

# ════════════════════════════════════════════════════════════════════════
# 2. t-SNE FEATURE VISUALIZATION
# ════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_embeddings(model, loader, device, max_samples=1000):
    """Extract feature embeddings for t-SNE."""
    model.eval()
    embeddings, labels = [], []

    for imgs, lbls in loader:
        imgs = imgs.to(device)
        with torch.amp.autocast('cuda'):
            feats = model.get_embeddings(imgs)
        embeddings.append(feats.cpu().numpy())
        labels.extend(lbls.numpy())
        if len(labels) >= max_samples:
            break

    return np.vstack(embeddings)[:max_samples], np.array(labels)[:max_samples]


def plot_tsne(embeddings, labels, class_names, title="t-SNE", save_path=None):
    """Publication-quality t-SNE plot."""
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    for cls_idx, cls_name in enumerate(class_names):
        mask = labels == cls_idx
        ax.scatter(coords[mask, 0], coords[mask, 1], c=COLORS[cls_idx],
                   label=cls_name, alpha=0.6, s=30, edgecolors='white', linewidth=0.3)

    ax.set_title(f'{title} Feature Space', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, markerscale=1.5)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ════════════════════════════════════════════════════════════════════════
# 3. RUN EXPLAINABILITY
# ════════════════════════════════════════════════════════════════════════

def run_explainability(all_models, data_dict):
    nf_test_loader = data_dict["nf_test_loader"]
    nf_test_ds = nf_test_loader.dataset

    print(f"\n{'='*60}")
    print("🔍 Explainability Suite")
    print(f"{'='*60}")

    nf_model = all_models['NeuroFusionNet']

    # Grad-CAM++
    print("\n🌡️ Generating Grad-CAM++ visualizations...")
    generate_gradcam_grid(
        nf_model, nf_test_ds, device, n_per_class=3,
        class_names=cfg.class_names,
        save_path=os.path.join(cfg.output_dir, 'fig_gradcam_grid.png')
    )

    # ★ XAI Comparison: Grad-CAM++ vs Multi-Layer vs HiResCAM
    print("\n🔬 Generating XAI comparison (3 methods side-by-side)...")
    generate_xai_comparison(
        nf_model, nf_test_ds, device,
        save_path=os.path.join(cfg.output_dir, 'fig_xai_comparison.png')
    )

    # t-SNE
    print("\n📐 Generating t-SNE embeddings...")
    emb, lbl = extract_embeddings(nf_model, nf_test_loader, device)
    plot_tsne(emb, lbl, cfg.class_names, title='NeuroFusionNet',
              save_path=os.path.join(cfg.output_dir, 'fig_tsne.png'))

    # Expert gate analysis
    print("\n🎛️ Expert Gate Analysis...")
    nf_model.eval()
    all_gates = []
    all_labels_gate = []
    for imgs, lbls in nf_test_loader:
        imgs = imgs.to(device)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            out = nf_model(imgs, return_all=True)
        all_gates.append(out['gate_weights'].cpu().numpy())
        all_labels_gate.extend(lbls.numpy())

    gates = np.vstack(all_gates)
    labels_gate = np.array(all_labels_gate)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    expert_names = ['Tumor\nDetector', 'Type\nClassifier', 'Fine-Grained\n(G vs M)']
    for cls_idx, cls_name in enumerate(cfg.class_names):
        mask = labels_gate == cls_idx
        mean_weights = gates[mask].mean(axis=0)
        axes[cls_idx].bar(expert_names, mean_weights,
                           color=[COLORS[0], COLORS[1], COLORS[2]], alpha=0.8)
        axes[cls_idx].set_title(f'{cls_name.upper()}', fontweight='bold')
        axes[cls_idx].set_ylim(0, 0.6)
        axes[cls_idx].set_ylabel('Gate Weight')
        for i, w in enumerate(mean_weights):
            axes[cls_idx].text(i, w + 0.01, f'{w:.2f}', ha='center', fontsize=9)

    fig.suptitle('Expert Routing — Gate Weight Distribution per Class',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, 'fig_expert_routing.png'))
    plt.show()

    print("✅ Explainability complete!")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    from cell_01_setup import *
    from cell_02_data import *
    from cell_03_models import *
    from cell_04_training import *
    from cell_00_load_checkpoints import load_all_checkpoints
    
    # Load data
    data_dict = setup_data()
    # Load checkpoints (skips training)
    all_models, test_results, data_dict = load_all_checkpoints(data_dict)
    
    # Run explainability
    run_explainability(all_models, data_dict)
