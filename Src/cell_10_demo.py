import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from cell_08_explainability import *

# %% Cell 10: ★ Interactive Demo — Gradio App for Professor Presentation
# ════════════════════════════════════════════════════════════════════════
# THREE-TIER CLINICAL DECISION SUPPORT DEMO:
#   ✅ AUTO:   High confidence → automated diagnosis + explanation
#   ⚠️ ASSIST: Medium confidence → top-2 candidates + uncertainty map
#   🚫 ABSTAIN: Low confidence → "Needs specialist review"
# ════════════════════════════════════════════════════════════════════════

import gradio as gr


# ════════════════════════════════════════════════════════════════════════
# 1. PREDICTION FUNCTION
# ════════════════════════════════════════════════════════════════════════

def predict_with_uncertainty(image, model, device, T=20):
    """Run MC Dropout inference on a single image.

    Returns:
        mean_probs: (C,) averaged softmax
        predictive_entropy: scalar
        mutual_information: scalar
        all_probs: (T, C) individual pass probabilities
    """
    # CRITICAL: enable_mc_dropout FORCES dropout layers to train mode
    # This must happen AFTER eval() or it will be overridden
    model.eval()
    model.enable_mc_dropout()  # This now immediately sets dropout to train()

    transform = get_transforms('test', 256)
    img_np = np.array(image)
    augmented = transform(image=img_np)
    img_tensor = augmented['image'].unsqueeze(0).to(device)

    amp_dtype = torch.bfloat16 if cfg.amp_dtype == 'bfloat16' else torch.float16
    all_probs = []

    with torch.no_grad():
        for _ in range(T):
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                logits = model(img_tensor)
                if isinstance(logits, dict):
                    logits = logits['logits']
                probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy()[0])

    model.disable_mc_dropout()

    all_probs = np.array(all_probs)  # (T, C)
    mean_probs = all_probs.mean(axis=0)

    # Verification: check MC Dropout is actually producing variance
    prob_std = all_probs.std(axis=0).mean()
    if prob_std < 1e-6:
        print("⚠️ WARNING: MC Dropout producing zero variance — dropout may not be active!")
    else:
        print(f"✅ MC Dropout variance: mean_std={prob_std:.6f} (T={T})")

    pred_entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10))
    exp_entropy = -np.mean(np.sum(all_probs * np.log(all_probs + 1e-10), axis=1))
    mutual_info = max(0.0, pred_entropy - exp_entropy)  # Clamp to 0+

    return mean_probs, pred_entropy, mutual_info, all_probs


def generate_gradcam_single(model, image, device):
    """Generate SHARP contour overlay using Grad-CAM + morphological post-processing.

    Pipeline:
      1. Grad-CAM++ heatmap (blurry)
      2. Otsu thresholding → binary mask
      3. Morphological cleanup (open + close)
      4. Draw contour on original image (segmentation-like)
    """
    import cv2

    try:
        model.eval()
        if hasattr(model, 'disable_mc_dropout'):
            model.disable_mc_dropout()

        transform = get_transforms('test', 256)
        img_np = np.array(image)
        augmented = transform(image=img_np)
        img_tensor = augmented['image'].unsqueeze(0).to(device)

        with torch.no_grad(), torch.amp.autocast('cuda'):
            logits = model(img_tensor)
            if isinstance(logits, dict):
                logits = logits['logits']
            pred_class = logits.argmax(1).item()
            pred_name = cfg.class_names[pred_class]

        # Generate Grad-CAM++ heatmap
        wrapped = ModelWrapper(model)
        target_layers = get_target_layers(model, 'local')
        if not target_layers:
            return None
        cam = GradCAMPlusPlus(model=wrapped, target_layers=target_layers)
        targets = [ClassifierOutputTarget(pred_class)]
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]
        cam.__del__()

        # Reconstruct RGB image
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        rgb = img_tensor[0].permute(1, 2, 0).cpu().numpy()
        rgb = (rgb * imagenet_std + imagenet_mean).clip(0, 1)
        rgb_uint8 = (rgb * 255).astype(np.uint8)

        # If "notumor" → no contour needed, just show light heatmap
        if pred_name == 'notumor':
            cam_img = show_cam_on_image(rgb.astype(np.float32), grayscale_cam * 0.3, use_rgb=True)
            # Add text
            cv2.putText(cam_img, "No Tumor Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            return cam_img

        # ── POST-PROCESSING: Heatmap → GrabCut-Refined Contour ──

        # Step 1: Moderate threshold (top 30% attention)
        cam_uint8 = (grayscale_cam * 255).astype(np.uint8)
        threshold_val = np.percentile(cam_uint8[cam_uint8 > 0], 70) if (cam_uint8 > 0).any() else 128
        _, binary = cv2.threshold(cam_uint8, int(threshold_val), 255, cv2.THRESH_BINARY)

        # Step 2: Clean up — open removes noise, close fills small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Step 3: Keep ONLY the largest connected region
        cnts_tmp, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts_tmp:
            largest = max(cnts_tmp, key=cv2.contourArea)
            binary = np.zeros_like(binary)
            cv2.drawContours(binary, [largest], -1, 255, cv2.FILLED)

        # Step 4: GrabCut refinement
        try:
            gc_mask = np.where(binary > 0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
            high_conf = cam_uint8 > np.percentile(cam_uint8[cam_uint8 > 0], 90) if (cam_uint8 > 0).any() else cam_uint8 > 200
            gc_mask[high_conf] = cv2.GC_FGD
            gc_mask[cam_uint8 == 0] = cv2.GC_BGD

            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            rgb_bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
            cv2.grabCut(rgb_bgr, gc_mask, None, bgd_model, fgd_model, 8, cv2.GC_INIT_WITH_MASK)

            refined_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

            # Single erosion to tighten, then smooth edges
            refined_mask = cv2.erode(refined_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
            refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
        except Exception:
            # GrabCut failed → use original binary mask
            refined_mask = binary

        # Step 4: Find contours on REFINED mask
        contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 5: Visualization
        result = rgb_uint8.copy()

        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            main_contour = contours[0]

            # Semi-transparent red fill
            overlay = result.copy()
            cv2.drawContours(overlay, [main_contour], -1, (255, 60, 60), -1)
            result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)

            # Cyan contour border (2px) — tight around tumor
            cv2.drawContours(result, [main_contour], -1, (0, 255, 255), 2)

            # ROI info
            area_pct = cv2.contourArea(main_contour) / (rgb_uint8.shape[0] * rgb_uint8.shape[1]) * 100
            cv2.putText(result, f"ROI: {area_pct:.1f}%", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Secondary regions
            for cnt in contours[1:3]:
                if cv2.contourArea(cnt) > 80:
                    cv2.drawContours(result, [cnt], -1, (255, 255, 0), 2)

        h = result.shape[0]
        cv2.putText(result, "GrabCut-Refined Localization", (5, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return result

    except Exception as e:
        print(f"Grad-CAM contour error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ════════════════════════════════════════════════════════════════════════
# 2. DEMO INTERFACE
# ════════════════════════════════════════════════════════════════════════

def run_demo(all_models, test_results, data_dict):
    test_df = data_dict["test_df"]
    # Get thresholds (from uncertainty analysis, or defaults)
    demo_theta_high = test_results.get('NeuroFusionNet', {}).get('thresholds', (0.85, 0.4))[0]
    demo_theta_low  = test_results.get('NeuroFusionNet', {}).get('thresholds', (0.85, 0.4))[1]

    TUMOR_INFO = {
        'glioma': {
            'name': 'Glioma (U thần kinh đệm)',
            'severity': '⚠️ Ác tính — Cần can thiệp khẩn',
            'desc': 'Glioma là loại u não ác tính phổ biến nhất, phát sinh từ tế bào thần kinh đệm. '
                    'Bao gồm nhiều phân loại phụ (astrocytoma, oligodendroglioma, glioblastoma). '
                    'Tiên lượng phụ thuộc vào grade và dấu ấn phân tử (IDH, 1p/19q).',
        },
        'meningioma': {
            'name': 'Meningioma (U màng não)',
            'severity': '✅ Thường lành tính — Theo dõi hoặc phẫu thuật',
            'desc': 'Meningioma phát triển từ màng não (meninges), thường lành tính (WHO grade I). '
                    'Phần lớn phát triển chậm và có thể theo dõi. Phẫu thuật thường có tiên lượng tốt.',
        },
        'notumor': {
            'name': 'Không phát hiện khối u',
            'severity': '✅ Bình thường',
            'desc': 'Ảnh MRI không cho thấy dấu hiệu khối u. Tuy nhiên, kết quả AI chỉ mang tính tham khảo. '
                    'Nên tham vấn bác sĩ chuyên khoa nếu có triệu chứng lâm sàng.',
        },
        'pituitary': {
            'name': 'Pituitary Tumor (U tuyến yên)',
            'severity': '⚠️ Cần đánh giá — Ảnh hưởng nội tiết',
            'desc': 'U tuyến yên phát triển ở vùng hố yên (sella turcica). '
                    'Có thể gây rối loạn nội tiết (prolactin, GH, ACTH). '
                    'Đa số lành tính nhưng cần theo dõi chức năng nội tiết.',
        },
    }


    def classify_brain_tumor(image):
        """Main classification function for Gradio demo."""
        if image is None:
            return "❌ Vui lòng upload ảnh MRI", None, None

        model = all_models.get('NeuroFusionNet')
        if model is None:
            return "❌ Model chưa được load", None, None

        # MC Dropout prediction
        mean_probs, pred_entropy, mutual_info, all_probs = predict_with_uncertainty(
            image, model, device, T=15
        )

        # Radiomics extraction (safe — may not be available)
        radiomics = None
        try:
            radiomics = compute_radiomics_for_image(model, image, device)
        except Exception:
            pass

        pred_class = mean_probs.argmax()
        confidence = mean_probs[pred_class]
        pred_name = cfg.class_names[pred_class]
        info = TUMOR_INFO.get(pred_name, {})

        # Determine tier (RISK-AWARE — key clinical innovation)
        risk_weights = {'glioma': 0.9, 'meningioma': 1.0, 'notumor': 1.4, 'pituitary': 1.0}
        risk_factor = risk_weights.get(pred_name, 1.0)
        adjusted_theta = min(demo_theta_high * risk_factor, 0.99)

        # Entropy gate: high uncertainty → never AUTO
        entropy_gate = pred_entropy > 1.0

        if confidence >= adjusted_theta and not entropy_gate:
            tier = "AUTO"
            tier_icon = "✅"
            tier_msg = "Độ tin cậy CAO — Chẩn đoán tự động"
        elif confidence >= demo_theta_low:
            tier = "ASSIST"
            tier_icon = "⚠️"
            tier_msg = "Cần bác sĩ xác nhận"
            if entropy_gate:
                tier_msg += " (entropy cao bất thường)"
            if risk_factor > 1.0:
                tier_msg += f" (risk-adjusted: θ={adjusted_theta:.2f})"
        else:
            tier = "ABSTAIN"
            tier_icon = "🚫"
            tier_msg = "Độ tin cậy THẤP — Cần chuyên gia đánh giá"

        # ── Compute doctor-friendly metrics ──
        # Agreement Score: how many MC passes agree on same class?
        mc_predictions = all_probs.argmax(axis=1)  # (T,)
        agreement_count = (mc_predictions == pred_class).sum()
        agreement_total = len(mc_predictions)
        agreement_pct = agreement_count / agreement_total * 100

        # Confusion Margin: gap between #1 and #2
        sorted_probs = np.sort(mean_probs)[::-1]
        confusion_margin = sorted_probs[0] - sorted_probs[1]
        sorted_idx = np.argsort(-mean_probs)

        # Expert Consensus (from NeuroFusionNet's 3 expert heads)
        expert_consensus = "N/A"
        try:
            model.eval()
            model.disable_mc_dropout()
            transform = get_transforms('test', 256)
            img_np = np.array(image)
            augmented = transform(image=img_np)
            img_t = augmented['image'].unsqueeze(0).to(device)
            amp_dtype = torch.bfloat16 if cfg.amp_dtype == 'bfloat16' else torch.float16
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=amp_dtype):
                out = model(img_t, return_all=True)
            # Expert 1: Tumor detector
            tumor_prob = F.softmax(out['tumor_logits'], dim=1)[0]
            has_tumor = tumor_prob[1].item() > 0.5
            # Expert 2: 4-class
            type_pred = out['logits'].argmax(1).item()
            # Expert 3: Glioma vs Meningioma
            fine_prob = F.softmax(out['fine_logits'], dim=1)[0]
            fine_pred = "Glioma" if fine_prob[0] > fine_prob[1] else "Meningioma"
            # Gate weights
            gate_w = out['gate_weights'][0].cpu().numpy()

            experts_agree = 0
            expert_details = []
            if has_tumor and pred_name != 'notumor':
                experts_agree += 1
                expert_details.append(f"Expert 1 (Phát hiện u): ✅ Có u ({tumor_prob[1]:.0%})")
            elif not has_tumor and pred_name == 'notumor':
                experts_agree += 1
                expert_details.append(f"Expert 1 (Phát hiện u): ✅ Không u ({tumor_prob[0]:.0%})")
            else:
                expert_details.append(f"Expert 1 (Phát hiện u): ⚠️ Xung đột (tumor={tumor_prob[1]:.0%})")

            if type_pred == pred_class:
                experts_agree += 1
                expert_details.append(f"Expert 2 (Phân loại): ✅ Đồng ý → {pred_name}")
            else:
                expert_details.append(f"Expert 2 (Phân loại): ⚠️ Khác ý → {cfg.class_names[type_pred]}")

            if pred_name in ['glioma', 'meningioma']:
                if (fine_pred.lower() == pred_name):
                    experts_agree += 1
                    expert_details.append(f"Expert 3 (Glioma/Menin): ✅ Đồng ý → {fine_pred} ({max(fine_prob):.0%})")
                else:
                    expert_details.append(f"Expert 3 (Glioma/Menin): ⚠️ → {fine_pred} ({max(fine_prob):.0%})")
            else:
                experts_agree += 1  # Expert 3 not applicable → no conflict
                expert_details.append(f"Expert 3 (Glioma/Menin): — Không áp dụng")

            expert_consensus = f"{experts_agree}/3 chuyên gia đồng thuận"
        except Exception as e:
            expert_details = [f"Lỗi: {e}"]
            experts_agree = 0

        # Clinical Confidence Level (composite → ★ stars)
        star_score = (
            (agreement_pct / 100) * 0.30 +       # MC agreement
            min(confusion_margin / 0.8, 1.0) * 0.25 +  # Margin clarity
            (experts_agree / 3) * 0.25 +           # Expert consensus
            confidence * 0.20                       # Raw confidence
        )
        stars_full = int(star_score * 5)
        stars_str = "★" * stars_full + "☆" * (5 - stars_full)

        # Clinical recommendation based on tier + prediction
        CLINICAL_RECS = {
            ('AUTO', 'glioma'): "Đề nghị: Tham vấn phẫu thuật thần kinh + MR Spectroscopy",
            ('AUTO', 'meningioma'): "Đề nghị: Theo dõi MRI 6 tháng hoặc phẫu thuật elective",
            ('AUTO', 'notumor'): "Đề nghị: Kết quả bình thường, tái khám nếu có triệu chứng",
            ('AUTO', 'pituitary'): "Đề nghị: Xét nghiệm nội tiết (prolactin, GH, ACTH, cortisol)",
            ('ASSIST', 'glioma'): "Đề nghị: MR Spectroscopy + Perfusion MRI + hội chẩn liên khoa",
            ('ASSIST', 'meningioma'): "Đề nghị: Đọc film bởi bác sĩ X-quang chuyên sâu",
            ('ASSIST', 'notumor'): "Đề nghị: Bác sĩ đọc lại — AI không chắc vùng bất thường",
            ('ASSIST', 'pituitary'): "Đề nghị: Xét nghiệm nội tiết + đánh giá thị trường",
            ('ABSTAIN', ''): "Đề nghị: Chuyển chuyên gia X-quang thần kinh đánh giá trực tiếp",
        }
        rec = CLINICAL_RECS.get((tier, pred_name), CLINICAL_RECS.get((tier, ''), ''))

        # ══════════════════════════════════════════════════════════════════
        # BUILD CLINICAL REPORT
        # ══════════════════════════════════════════════════════════════════

        report_lines = [
            f"# {tier_icon} Chế độ: {tier} — {tier_msg}",
            f"",
            f"## 🏥 Kết quả Chẩn đoán",
            f"- **Chẩn đoán**: {info.get('name', pred_name)}",
            f"- **Mức độ**: {info.get('severity', 'N/A')}",
            f"",
            f"## {stars_str} Mức độ Tin cậy Lâm sàng ({star_score:.0%})",
            f"",
            f"| Chỉ số | Kết quả | Ý nghĩa |",
            f"|---|---|---|",
            f"| **Đồng thuận AI** | {agreement_count}/{agreement_total} lần ({agreement_pct:.0f}%) | {'Rất nhất quán' if agreement_pct >= 90 else 'Có dao động' if agreement_pct >= 70 else 'Không ổn định'} |",
            f"| **Cách biệt Top-2** | {confusion_margin:.0%} | {'Rõ ràng' if confusion_margin > 0.5 else 'Mơ hồ' if confusion_margin > 0.2 else 'Rất mơ hồ'} |",
            f"| **Hội chẩn Expert** | {expert_consensus} | {'Đồng thuận' if experts_agree == 3 else 'Có xung đột'} |",
            f"| **Confidence** | {confidence:.1%} | Xác suất dự đoán |",
        ]

        # Detail: MC Agreement interpretation
        if agreement_pct >= 90:
            report_lines.append(f"\n> ✅ **{agreement_count}/{agreement_total}** lần phân tích cho cùng kết quả — AI rất nhất quán")
        elif agreement_pct >= 70:
            report_lines.append(f"\n> ⚠️ **{agreement_count}/{agreement_total}** lần đồng ý — có {agreement_total - agreement_count} lần AI chọn khác")
        else:
            report_lines.append(f"\n> 🚫 Chỉ **{agreement_count}/{agreement_total}** lần đồng ý — AI đang rất BỐI RỐI")

        # Detail: Top-2 differential
        report_lines.extend([
            f"",
            f"## 🔬 Chẩn đoán Phân biệt (Differential Diagnosis)",
        ])
        for rank, idx in enumerate(sorted_idx[:3]):
            pct = mean_probs[idx]
            marker = " ← chẩn đoán chính" if idx == pred_class else ""
            bar_len = int(pct * 25)
            bar = "█" * bar_len + "░" * (25 - bar_len)
            report_lines.append(f"  {rank+1}. **{cfg.class_names[idx]}** {bar} {pct:.1%}{marker}")

        # Detail: Expert consensus
        report_lines.extend([
            f"",
            f"## 👥 Hội chẩn 3 Expert Heads",
        ])
        for detail in expert_details:
            report_lines.append(f"  - {detail}")

        # Detail: ASSIST/ABSTAIN specific
        if tier == "ASSIST":
            report_lines.extend([
                f"",
                f"## ⚠️ Chế độ ASSIST — Cần bác sĩ xác nhận",
                f"  Top-1: **{cfg.class_names[sorted_idx[0]]}** ({mean_probs[sorted_idx[0]]:.1%})",
                f"  Top-2: **{cfg.class_names[sorted_idx[1]]}** ({mean_probs[sorted_idx[1]]:.1%})",
                f"  Cách biệt chỉ {confusion_margin:.0%} — chưa đủ để kết luận.",
            ])
        elif tier == "ABSTAIN":
            report_lines.extend([
                f"",
                f"## 🚫 Chế độ ABSTAIN",
                f"  AI không đủ tự tin để đưa ra chẩn đoán.",
            ])

        # Radiomics section
        if radiomics is not None:
            report_lines.extend([
                f"",
                f"## 📐 Morphological Descriptors (Attention-Driven)*",
                f"| Đặc trưng | Giá trị | Ý nghĩa |",
                f"|---|---|---|",
                f"| Entropy | {radiomics.get('intensity_entropy', 0):.2f} | {'Heterogeneous → gợi ý grade cao' if radiomics.get('intensity_entropy', 0) > 4.0 else 'Đồng nhất → gợi ý lành tính'} |",
                f"| GLCM Contrast | {radiomics.get('glcm_contrast', 0):.1f} | {'Kết cấu phức tạp' if radiomics.get('glcm_contrast', 0) > 30 else 'Kết cấu mịn'} |",
                f"| Irregularity | {radiomics.get('shape_irregularity', 0):.2f} | {'Bờ nham nhở → xâm lấn' if radiomics.get('shape_irregularity', 0) > 1.5 else 'Bờ đều đặn'} |",
                f"",
                f"*\\*Tính từ Grad-CAM attention mask, KHÔNG phải từ segmentation thật (xấp xỉ).*",
            ])
            clinical_interp = radiomics.get('clinical_interpretation', '')
            if clinical_interp:
                report_lines.extend([f"", f"  {clinical_interp}"])

        # Clinical recommendation
        report_lines.extend([
            f"",
            f"## 📋 Đề nghị Lâm sàng",
            f"  **{rec}**",
            f"",
            f"## Mô tả",
            f"  {info.get('desc', 'N/A')}",
            f"",
            f"---",
            f"*⚕️ Kết quả AI chỉ mang tính hỗ trợ. Mọi quyết định lâm sàng*",
            f"*cần được thực hiện bởi bác sĩ chuyên khoa.*",
        ])

        report = "\n".join(report_lines)

        # Grad-CAM (safe fallback)
        gradcam_img = None
        try:
            gradcam_img = generate_gradcam_single(model, image, device)
        except Exception as e:
            print(f"Grad-CAM error: {e}")

        # Probability bar chart
        CLASS_COLORS = {
            'glioma': '#E63946', 'meningioma': '#457B9D',
            'notumor': '#2A9D8F', 'pituitary': '#E9C46A'
        }
        prob_path = None
        try:
            fig, ax = plt.subplots(figsize=(6, 3))
            bar_colors = [CLASS_COLORS.get(n, '#888') for n in cfg.class_names]
            bars = ax.barh(cfg.class_names, mean_probs, color=bar_colors,
                            edgecolor='black', linewidth=0.5, alpha=0.8)
            for bar, p in zip(bars, mean_probs):
                ax.text(p + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{p:.1%}', va='center', fontsize=10, fontweight='bold')
            ax.set_xlim(0, 1.15)
            ax.set_xlabel('Probability')
            ax.set_title(f'Prediction: {info.get("name", pred_name)} [{tier}]',
                         fontweight='bold', fontsize=12)
            ax.axvline(x=demo_theta_high, color='green', linestyle='--', alpha=0.5,
                       label=f'θ_auto={demo_theta_high:.2f}')
            ax.axvline(x=demo_theta_low, color='red', linestyle='--', alpha=0.5,
                       label=f'θ_abstain={demo_theta_low:.2f}')
            ax.legend(fontsize=8, loc='lower right')
            plt.tight_layout()
            prob_path = os.path.join(cfg.output_dir, 'demo_prob_chart.png')
            plt.savefig(prob_path)
            plt.close()
        except Exception as e:
            print(f"Chart error: {e}")
            plt.close('all')

        return report, gradcam_img, prob_path


    # ════════════════════════════════════════════════════════════════════════
    # 3. BUILD GRADIO APP
    # ════════════════════════════════════════════════════════════════════════

    # Get sample images from test set
    sample_examples = []
    for cls_name in cfg.class_names:
        cls_imgs = test_df[test_df['label'] == cls_name].sample(1, random_state=42)
        for _, row in cls_imgs.iterrows():
            sample_examples.append(row['path'])

    with gr.Blocks(
        title="TrustMed-AI: Brain Tumor Classification",
        theme=gr.themes.Soft(),
        css="""
        .main-title { text-align: center; font-size: 2em; font-weight: bold;
                      color: #1a1a2e; margin-bottom: 10px; }
        .subtitle { text-align: center; color: #666; margin-bottom: 20px; }
        """
    ) as demo:

        gr.HTML("""
        <div class="main-title">🧠 TrustMed-AI</div>
        <div class="subtitle">
            Trustworthy Brain Tumor MRI Classification<br>
            <em>Uncertainty-Aware Hierarchical Expert Voting with Selective Prediction</em>
        </div>
        <div style="text-align:center; margin-bottom:15px;">
            <span style="background:#2A9D8F;color:white;padding:3px 8px;border-radius:4px;">✅ AUTO</span>
            <span style="background:#E9C46A;color:black;padding:3px 8px;border-radius:4px;">⚠️ ASSIST</span>
            <span style="background:#94A3B8;color:white;padding:3px 8px;border-radius:4px;">🚫 ABSTAIN</span>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="📤 Upload Brain MRI")
                classify_btn = gr.Button("🔬 Analyze", variant="primary", size="lg")
                gr.Examples(
                    examples=sample_examples,
                    inputs=input_image,
                    label="📋 Sample MRI Images",
                )

            with gr.Column(scale=1):
                output_report = gr.Markdown(label="📋 Clinical Report")
                with gr.Row():
                    output_gradcam = gr.Image(label="🎯 Tumor Localization")
                    output_probs = gr.Image(label="📊 Probability Distribution")

        classify_btn.click(
            fn=classify_brain_tumor,
            inputs=input_image,
            outputs=[output_report, output_gradcam, output_probs],
        )

        gr.HTML("""
        <div style="text-align:center; margin-top:20px; color:#888; font-size:0.9em;">
            ⚕️ Research prototype — Not for clinical use without physician supervision<br>
            NeuroFusionNet: Swin-V2-S + ConvNeXt-Base | MC Dropout (T=15) | Selective Prediction
        </div>
        """)

    print("\n🚀 Launching demo...")
    demo.launch(share=True, debug=False)

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
    
    # Run interactive demo
    run_demo(all_models, test_results, data_dict)
