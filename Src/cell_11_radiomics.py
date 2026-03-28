import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from cell_08_explainability import *

# %% Cell 11: ★ Attention-Driven Morphological Descriptors
# ════════════════════════════════════════════════════════════════════════
# CLINICAL INSIGHT:
#
# A classification label alone is USELESS to a clinician. When a
# radiologist reads brain MRI, they extract quantitative descriptors:
#   - Texture heterogeneity → correlates with tumor grade
#   - Shape irregularity → correlates with infiltration/malignancy
#   - Intensity statistics → correlates with necrosis, edema, cellularity
#
# These features are what radiologists spend YEARS learning to evaluate
# by eye. AI can compute them in milliseconds with higher precision.
#
# KEY INNOVATION: We use Grad-CAM attention map as a soft tumor mask
# to extract morphological features WITHOUT requiring ground-truth
# segmentation masks (which this dataset doesn't provide).
#
# ⚠️ TERMINOLOGY NOTE (for paper):
#   These are NOT standard "Radiomics" as defined by IBSI (Image
#   Biomarker Standardisation Initiative). IBSI radiomics require
#   precise tumor segmentation masks (manual or semi-automatic).
#   Our features use Grad-CAM attention as a soft proxy mask →
#   they are "Attention-Driven Morphological Descriptors" or
#   "Pseudo-Radiomics". The function names retain 'radiomics'
#   for backward compatibility only.
# ════════════════════════════════════════════════════════════════════════

from skimage.feature import graycomatrix, graycoprops
from scipy import ndimage
from scipy.stats import skew, kurtosis as sp_kurtosis


# ════════════════════════════════════════════════════════════════════════
# 1. GLCM TEXTURE FEATURES
# ════════════════════════════════════════════════════════════════════════
# Gray-Level Co-occurrence Matrix captures spatial relationships between
# pixel intensities — THE gold standard for tumor texture analysis.
#
# Clinical meaning:
#   Contrast     ↑ = more intensity variation = more heterogeneous tumor
#   Correlation  ↑ = more structured/organized tissue
#   Energy       ↑ = more uniform (homogeneous) tissue
#   Homogeneity  ↑ = less variation between adjacent pixels
#
#   High-grade glioma: HIGH contrast, LOW energy (chaotic interior)
#   Meningioma:        LOW contrast, HIGH energy (uniform, well-defined)
# ════════════════════════════════════════════════════════════════════════

def extract_glcm_features(gray_roi, num_levels=32):
    """Extract GLCM texture features from a grayscale ROI.

    Uses 4 directions (0°, 45°, 90°, 135°) and averages — rotation
    invariant, which is essential because tumor orientation on MRI
    depends on patient head position, not biology.

    Args:
        gray_roi: 2D grayscale image (masked tumor region)
        num_levels: quantization levels (32 is standard for radiomics)

    Returns:
        dict of GLCM features with clinical interpretation
    """
    if gray_roi.size == 0 or gray_roi.max() == gray_roi.min():
        return {
            'glcm_contrast': 0.0, 'glcm_correlation': 0.0,
            'glcm_energy': 1.0, 'glcm_homogeneity': 1.0,
            'glcm_dissimilarity': 0.0,
        }

    # Quantize to num_levels
    roi_q = np.clip(gray_roi, 0, 255)
    roi_q = ((roi_q / 255.0) * (num_levels - 1)).astype(np.uint8)

    # 4 directions, distance=1 (adjacent pixels)
    glcm = graycomatrix(
        roi_q, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=num_levels, symmetric=True, normed=True
    )

    features = {}
    for prop in ['contrast', 'correlation', 'energy', 'homogeneity', 'dissimilarity']:
        values = graycoprops(glcm, prop)
        features[f'glcm_{prop}'] = float(values.mean())

    return features


# ════════════════════════════════════════════════════════════════════════
# 2. FIRST-ORDER INTENSITY FEATURES
# ════════════════════════════════════════════════════════════════════════
# Signal intensity distribution within the tumor region.
#
# Clinical meaning:
#   Mean intensity  → overall signal character of the tumor
#   Std deviation   → intra-tumoral variability
#   Skewness        ↑ = tail towards bright → possible enhancement/hemorrhage
#   Kurtosis        ↑ = heavy tails → extreme intensity values (necrosis)
#   Entropy         ↑ = more disorder → heterogeneous → higher grade
#
#   Professor's note: Entropy is the single most predictive radiomics
#   feature for tumor grade (validated across 50+ studies since 2012).
# ════════════════════════════════════════════════════════════════════════

def extract_intensity_features(gray_roi):
    """Extract first-order intensity statistics from tumor ROI.

    These features capture what radiologists describe as:
    - "homogeneously enhancing" → low entropy, low std
    - "heterogeneously enhancing" → high entropy, high std
    - "necrotic center" → high kurtosis (extreme values)
    """
    if gray_roi.size == 0:
        return {
            'intensity_mean': 0.0, 'intensity_std': 0.0,
            'intensity_skewness': 0.0, 'intensity_kurtosis': 0.0,
            'intensity_entropy': 0.0, 'intensity_range': 0.0,
            'intensity_p10': 0.0, 'intensity_p90': 0.0,
        }

    pixels = gray_roi.flatten().astype(float)

    # Entropy (Shannon)
    hist, _ = np.histogram(pixels, bins=64, range=(0, 255), density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist + 1e-10))

    features = {
        'intensity_mean': float(np.mean(pixels)),
        'intensity_std': float(np.std(pixels)),
        'intensity_skewness': float(skew(pixels)) if len(pixels) > 2 else 0.0,
        'intensity_kurtosis': float(sp_kurtosis(pixels)) if len(pixels) > 2 else 0.0,
        'intensity_entropy': float(entropy),
        'intensity_range': float(np.ptp(pixels)),
        'intensity_p10': float(np.percentile(pixels, 10)),
        'intensity_p90': float(np.percentile(pixels, 90)),
    }
    return features


# ════════════════════════════════════════════════════════════════════════
# 3. SHAPE FEATURES (from Grad-CAM mask)
# ════════════════════════════════════════════════════════════════════════
# Shape of the attention region approximates tumor morphology.
#
# Clinical meaning:
#   Irregularity   ↑ = irregular borders → infiltrative → malignant
#   Solidity       ↑ = compact shape (solid = meningioma-like)
#   Eccentricity   ↑ = elongated (non-spherical = unusual growth)
#   Relative area  ↑ = large tumor relative to brain = worse prognosis
#
#   Meningioma: smooth, round, well-defined → irregularity ≈ 1.0
#   Glioma (HGG): irregular, infiltrative → irregularity > 1.5
# ════════════════════════════════════════════════════════════════════════

def extract_shape_features(binary_mask):
    """Extract shape descriptors from the binary tumor mask.

    Uses connected component analysis — same approach as
    clinical tumor volumetry software (BrainLab, 3D Slicer).
    """
    if binary_mask.sum() == 0:
        return {
            'shape_area_ratio': 0.0, 'shape_perimeter': 0.0,
            'shape_irregularity': 1.0, 'shape_solidity': 1.0,
            'shape_eccentricity': 0.0, 'shape_n_components': 0,
        }

    # Connected components
    labeled, n_components = ndimage.label(binary_mask)
    total_pixels = binary_mask.size
    tumor_pixels = binary_mask.sum()

    # Use largest component
    if n_components > 0:
        component_sizes = ndimage.sum(binary_mask, labeled, range(1, n_components + 1))
        largest_idx = np.argmax(component_sizes) + 1
        largest_mask = (labeled == largest_idx).astype(np.uint8)
    else:
        largest_mask = binary_mask.astype(np.uint8)

    area = largest_mask.sum()

    # Perimeter (count boundary pixels)
    eroded = ndimage.binary_erosion(largest_mask)
    perimeter = float((largest_mask.astype(int) - eroded.astype(int)).sum())

    # Irregularity index: perimeter² / (4π × area)
    # = 1.0 for perfect circle, > 1.0 for irregular shapes
    if area > 0:
        irregularity = (perimeter ** 2) / (4 * np.pi * area)
    else:
        irregularity = 1.0

    # Solidity: area / convex_hull_area (approximation)
    # Using bounding box as rough convex hull approximation
    rows = np.any(largest_mask, axis=1)
    cols = np.any(largest_mask, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        bbox_area = max((rmax - rmin + 1) * (cmax - cmin + 1), 1)
        solidity = area / bbox_area
        # Eccentricity: ratio of bbox dimensions
        h, w = rmax - rmin + 1, cmax - cmin + 1
        eccentricity = max(h, w) / max(min(h, w), 1)
    else:
        solidity = 1.0
        eccentricity = 1.0

    features = {
        'shape_area_ratio': float(tumor_pixels / total_pixels),  # % of brain
        'shape_perimeter': perimeter,
        'shape_irregularity': float(irregularity),
        'shape_solidity': float(solidity),
        'shape_eccentricity': float(eccentricity),
        'shape_n_components': int(n_components),
    }
    return features


# ════════════════════════════════════════════════════════════════════════
# 4. FULL RADIOMICS EXTRACTION PIPELINE
# ════════════════════════════════════════════════════════════════════════

def extract_radiomics_from_gradcam(image_rgb, gradcam_map, threshold=0.3):
    """Extract all radiomics features using Grad-CAM as tumor localization.

    Args:
        image_rgb: (H, W, 3) RGB image (normalized 0-1)
        gradcam_map: (H, W) Grad-CAM attention map (0-1)
        threshold: attention threshold for binary mask

    Returns:
        dict with all radiomics features + clinical interpretation
    """
    # Convert to grayscale
    gray = np.mean(image_rgb, axis=2)
    gray_uint8 = (gray * 255).astype(np.uint8)

    # Create binary mask from Grad-CAM
    binary_mask = (gradcam_map >= threshold).astype(np.uint8)

    # Apply mask to extract tumor-only pixels
    tumor_region = gray_uint8 * binary_mask

    # Extract non-zero ROI for GLCM (need rectangular patch)
    if binary_mask.sum() > 100:  # minimum meaningful size
        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        roi = gray_uint8[rmin:rmax+1, cmin:cmax+1]
        roi_mask = binary_mask[rmin:rmax+1, cmin:cmax+1]
        # Only keep tumor pixels for intensity
        tumor_pixels = gray_uint8[binary_mask > 0]
    else:
        roi = gray_uint8
        roi_mask = np.ones_like(gray_uint8)
        tumor_pixels = gray_uint8.flatten()

    # Extract features
    glcm_feats = extract_glcm_features(roi)
    intensity_feats = extract_intensity_features(tumor_pixels)
    shape_feats = extract_shape_features(binary_mask)

    # Combine all
    all_features = {}
    all_features.update(glcm_feats)
    all_features.update(intensity_feats)
    all_features.update(shape_feats)

    # Add clinical interpretation
    all_features['clinical_interpretation'] = interpret_radiomics(all_features)

    return all_features


# ════════════════════════════════════════════════════════════════════════
# 5. CLINICAL INTERPRETATION ENGINE
# ════════════════════════════════════════════════════════════════════════
# This is what separates an AI engineer from someone with clinical
# understanding. Raw numbers are meaningless — interpretation matters.

def interpret_radiomics(features):
    """Generate clinical interpretation of radiomics features.

    Based on published literature:
    - Kickingereder et al., 2016 (Radiology): GLCM for glioma grading
    - Kang et al., 2018 (AJNR): Texture features in meningioma subtypes
    - Tian et al., 2020 (EBioMedicine): Radiomics for brain tumor prognosis
    """
    interpretation = []

    # ── Entropy interpretation ──
    entropy = features.get('intensity_entropy', 0)
    if entropy > 4.5:
        interpretation.append(
            "🔴 Entropy CAO ({:.2f}) — Vùng u rất KHÔNG đồng nhất (heterogeneous). "
            "Trong glioma, entropy cao tương quan với grade cao (III-IV). "
            "Có thể chứa vùng hoại tử, xuất huyết, hoặc tăng sinh mạch hỗn loạn.".format(entropy)
        )
    elif entropy > 3.5:
        interpretation.append(
            "🟡 Entropy TRUNG BÌNH ({:.2f}) — Mức heterogeneity trung bình. "
            "Phù hợp với u có cấu trúc tổ chức một phần.".format(entropy)
        )
    else:
        interpretation.append(
            "🟢 Entropy THẤP ({:.2f}) — Vùng u tương đối ĐỒNG NHẤT. "
            "Thường thấy ở meningioma (WHO Grade I) hoặc u lành tính.".format(entropy)
        )

    # ── GLCM Contrast interpretation ──
    contrast = features.get('glcm_contrast', 0)
    energy = features.get('glcm_energy', 0)

    if contrast > 50:
        interpretation.append(
            "🔴 GLCM Contrast CAO ({:.1f}) — Sự biến đổi cường độ mạnh "
            "giữa các pixel liền kề → kết cấu phức tạp, gợi ý ác tính.".format(contrast)
        )
    elif contrast < 10:
        interpretation.append(
            "🟢 GLCM Contrast THẤP ({:.1f}) — Kết cấu mịn, đồng nhất → "
            "thường gặp ở u lành tính hoặc não bình thường.".format(contrast)
        )

    if energy > 0.3:
        interpretation.append(
            "🟢 GLCM Energy CAO ({:.3f}) — Vùng u có kết cấu uniform, "
            "ít biến đổi → đặc trưng meningioma.".format(energy)
        )

    # ── Shape interpretation ──
    irregularity = features.get('shape_irregularity', 1.0)
    if irregularity > 2.0:
        interpretation.append(
            "🔴 Irregularity CAO ({:.2f}) — Bờ u KHÔNG ĐỀU, nham nhở. "
            "Dấu hiệu xâm lấn (infiltration) → gợi ý glioma ác tính. "
            "Meningioma thường có bờ đều (irregularity < 1.5).".format(irregularity)
        )
    elif irregularity < 1.3:
        interpretation.append(
            "🟢 Irregularity THẤP ({:.2f}) — Bờ u ĐỀU ĐẶN, tròn. "
            "Đặc trưng u lành tính (meningioma, schwannoma).".format(irregularity)
        )

    # ── Area ratio ──
    area_ratio = features.get('shape_area_ratio', 0)
    if area_ratio > 0.2:
        interpretation.append(
            "⚠️ Kích thước LỚN ({:.1%} diện tích não) — "
            "Cần đánh giá hiệu ứng khối (mass effect), lệch đường giữa.".format(area_ratio)
        )

    # ── Heterogeneity score (composite) ──
    hetero_score = (
        min(entropy / 5.0, 1.0) * 0.4 +
        min(contrast / 100.0, 1.0) * 0.3 +
        min(features.get('intensity_std', 0) / 60.0, 1.0) * 0.3
    )
    interpretation.append(
        "\n📊 **Heterogeneity Score: {:.2f}/1.00** ".format(hetero_score) +
        ("(CAO → gợi ý high-grade)" if hetero_score > 0.5
         else "(THẤP → gợi ý low-grade/lành tính)")
    )

    return "\n".join(interpretation)


# ════════════════════════════════════════════════════════════════════════
# 6. FULL RADIOMICS GUI FUNCTION (for Grad-CAM integration)
# ════════════════════════════════════════════════════════════════════════

def compute_radiomics_for_image(model, image_pil, device):
    """End-to-end: image → Grad-CAM → radiomics features.

    Used by the Gradio demo to add radiomics to the clinical report.
    """
    try:
        model.eval()
        model.disable_mc_dropout()

        transform = get_transforms('test', 256)
        img_np = np.array(image_pil)
        augmented = transform(image=img_np)
        img_tensor = augmented['image'].unsqueeze(0).to(device)

        # Get Grad-CAM map
        wrapped = ModelWrapper(model)
        target_layers = get_target_layers(model, 'local')
        if not target_layers:
            return None

        cam = GradCAMPlusPlus(model=wrapped, target_layers=target_layers)

        with torch.amp.autocast('cuda'):
            logits = model(img_tensor)
            if isinstance(logits, dict):
                logits = logits['logits']
            pred_class = logits.argmax(1).item()

        targets = [ClassifierOutputTarget(pred_class)]
        gradcam_map = cam(input_tensor=img_tensor, targets=targets)[0]

        # Reconstruct RGB from tensor
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        rgb = img_tensor[0].permute(1, 2, 0).cpu().numpy()
        rgb = (rgb * imagenet_std + imagenet_mean).clip(0, 1)

        # Extract radiomics
        features = extract_radiomics_from_gradcam(
            rgb, gradcam_map, threshold=0.3
        )

        cam.__del__()
        return features

    except Exception as e:
        print(f"Radiomics error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ════════════════════════════════════════════════════════════════════════
# 7. BATCH RADIOMICS ANALYSIS (for paper figures)
# ════════════════════════════════════════════════════════════════════════

def batch_radiomics_analysis(model, dataset, test_df, device, n_samples=200):
    """Extract radiomics for many samples → analyze per-class patterns."""
    model.eval()
    model.disable_mc_dropout()

    wrapped = ModelWrapper(model)
    target_layers = get_target_layers(model, 'local')
    if not target_layers:
        print("⚠️ Cannot get target layers")
        return None

    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == 'bfloat16' else torch.float16

    results = []
    n = min(n_samples, len(dataset))
    indices = np.random.RandomState(42).choice(len(dataset), n, replace=False)

    cam = GradCAMPlusPlus(model=wrapped, target_layers=target_layers)

    for i in tqdm(indices, desc="Extracting radiomics", leave=False):
        try:
            img_tensor, label = dataset[i]
            img_input = img_tensor.unsqueeze(0).to(device)

            with torch.amp.autocast('cuda', dtype=amp_dtype):
                logits = model(img_input)
                if isinstance(logits, dict):
                    logits = logits['logits']
                pred = logits.argmax(1).item()

            targets = [ClassifierOutputTarget(pred)]
            gradcam_map = cam(input_tensor=img_input, targets=targets)[0]

            rgb = img_tensor.permute(1, 2, 0).numpy()
            rgb = (rgb * imagenet_std + imagenet_mean).clip(0, 1)

            feats = extract_radiomics_from_gradcam(rgb, gradcam_map)
            feats['true_label'] = cfg.class_names[label]
            feats['pred_label'] = cfg.class_names[pred]
            feats['correct'] = (label == pred)
            results.append(feats)

        except Exception:
            continue

    cam.__del__()
    return pd.DataFrame(results)


# ════════════════════════════════════════════════════════════════════════
# 8. RUN RADIOMICS ANALYSIS
# ════════════════════════════════════════════════════════════════════════

def run_radiomics(all_models, data_dict):
    test_df = data_dict["test_df"]
    nf_test_loader = data_dict["nf_test_loader"]
    nf_test_ds = nf_test_loader.dataset
    print(f"\n{'='*60}")
    print("📐 Attention-Driven Morphological Feature Extraction")
    print("   (Pseudo-Radiomics from Grad-CAM attention masks)")
    print(f"{'='*60}")

    nf_model = all_models['NeuroFusionNet']

    print("\n⏳ Extracting radiomics features from test set (200 samples)...")
    radiomics_df = batch_radiomics_analysis(nf_model, nf_test_ds, test_df, device, n_samples=200)

    if radiomics_df is not None and len(radiomics_df) > 0:
        print(f"\n✅ Extracted radiomics from {len(radiomics_df)} images")

        # Per-class radiomics statistics
        print(f"\n{'='*70}")
        print("📊 Radiomics per Class (Mean ± Std):")
        print(f"{'='*70}")

        key_features = [
            'intensity_entropy', 'glcm_contrast', 'glcm_energy',
            'shape_irregularity', 'intensity_std'
        ]

        for feat in key_features:
            print(f"\n  {feat}:")
            for cls_name in cfg.class_names:
                sub = radiomics_df[radiomics_df['true_label'] == cls_name][feat]
                if len(sub) > 0:
                    print(f"    {cls_name:<12}: {sub.mean():.3f} ± {sub.std():.3f}")

        # ── Publication figure: Radiomics violin plots ──
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        plot_features = [
            ('intensity_entropy', 'Signal Entropy\n(Heterogeneity)'),
            ('glcm_contrast', 'GLCM Contrast\n(Texture Complexity)'),
            ('glcm_energy', 'GLCM Energy\n(Texture Uniformity)'),
            ('shape_irregularity', 'Shape Irregularity\n(Border Regularity)'),
            ('intensity_std', 'Intensity Std\n(Signal Variation)'),
            ('shape_area_ratio', 'Relative Area\n(Tumor Size)')
        ]

        for idx, (feat, title) in enumerate(plot_features):
            ax = axes[idx // 3, idx % 3]
            data_per_class = [
                radiomics_df[radiomics_df['true_label'] == c][feat].values
                for c in cfg.class_names
            ]
            parts = ax.violinplot(data_per_class, showmeans=True, showmedians=True)
            for pc, color in zip(parts['bodies'], [COLORS[i] for i in range(4)]):
                pc.set_facecolor(color)
                pc.set_alpha(0.6)
            ax.set_xticks(range(1, 5))
            ax.set_xticklabels(cfg.class_names, fontsize=10)
            ax.set_title(title, fontsize=12, fontweight='bold')

        fig.suptitle('Radiomics Feature Distribution per Tumor Class',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.output_dir, 'fig_radiomics_violin.png'))
        plt.show()

        # ── Correlation between radiomics and prediction correctness ──
        print("\n📊 Radiomics → Prediction Quality Correlation:")
        for feat in key_features:
            correct_vals = radiomics_df[radiomics_df['correct']][feat]
            incorrect_vals = radiomics_df[~radiomics_df['correct']][feat]
            if len(incorrect_vals) > 0:
                print(f"  {feat}: correct={correct_vals.mean():.3f} vs "
                      f"incorrect={incorrect_vals.mean():.3f}")

        print("\n✅ Radiomics analysis complete!")
    else:
        print("⚠️ Radiomics extraction failed or returned empty results")

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
    
    # Run radiomics extraction
    run_radiomics(all_models, data_dict)
