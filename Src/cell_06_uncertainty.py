import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from cell_00_load_checkpoints import *

# %% Cell 6: ★ Uncertainty Engine + Selective Prediction
# ════════════════════════════════════════════════════════════════════════
# CORE NOVEL CONTRIBUTION: "A medical AI that knows when it doesn't know"
#
# Methods:
#   1. MC Dropout — T forward passes with active dropout
#   2. Calibration — ECE, reliability diagram, temperature scaling
#   3. Selective Prediction — Coverage-Accuracy trade-off
#   4. AURC — Area Under Risk-Coverage curve
# ════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
# 1. MC DROPOUT INFERENCE
# ════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def mc_dropout_inference(model, loader, device, T=30):
    """Monte Carlo Dropout: run T stochastic forward passes.

    Returns:
        labels: (N,) GT labels
        mean_probs: (N, C) mean softmax probabilities
        all_probs: (T, N, C) individual probabilities per pass
        predictive_entropy: (N,) total uncertainty
        mutual_information: (N,) epistemic (model) uncertainty
    """
    model.eval()
    model.enable_mc_dropout()

    amp_dtype = torch.bfloat16 if cfg.amp_dtype == 'bfloat16' else torch.float16
    all_labels = []
    all_passes = []  # List of T arrays, each (N, C)

    # Collect labels once
    for images, labels in loader:
        all_labels.extend(labels.numpy())
    all_labels = np.array(all_labels)

    for t in tqdm(range(T), desc="MC Dropout", leave=False):
        probs_t = []
        for images, _ in loader:
            images = images.to(device)
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                logits = model(images)
                if isinstance(logits, dict):
                    logits = logits['logits']
                probs = F.softmax(logits, dim=1)
            probs_t.append(probs.cpu().numpy())
        all_passes.append(np.concatenate(probs_t, axis=0))

    model.disable_mc_dropout()

    # Stack: (T, N, C)
    all_probs = np.stack(all_passes, axis=0)
    mean_probs = all_probs.mean(axis=0)  # (N, C)

    # Predictive entropy: H[E[p]] = -Σ p̄ log p̄
    predictive_entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)

    # Expected entropy: E[H[p]] = -1/T Σ_t Σ_c p_tc log p_tc
    expected_entropy = -np.mean(
        np.sum(all_probs * np.log(all_probs + 1e-10), axis=2), axis=0
    )

    # Mutual Information (epistemic uncertainty): I = H[E[p]] - E[H[p]]
    mutual_information = predictive_entropy - expected_entropy

    return {
        'labels': all_labels,
        'mean_probs': mean_probs,
        'all_probs': all_probs,
        'preds': mean_probs.argmax(axis=1),
        'max_prob': mean_probs.max(axis=1),
        'predictive_entropy': predictive_entropy,
        'mutual_information': mutual_information,
    }


# ════════════════════════════════════════════════════════════════════════
# 2. CALIBRATION METRICS & VISUALIZATION
# ════════════════════════════════════════════════════════════════════════

def compute_ece(probs, labels, n_bins=15):
    """Expected Calibration Error."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            bin_data.append((0, 0, 0))
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (n_in_bin / len(labels)) * abs(bin_acc - bin_conf)
        bin_data.append((bin_conf, bin_acc, n_in_bin))

    return ece, bin_data


def compute_brier_score(probs, labels, num_classes=4):
    """Multi-class Brier score."""
    one_hot = np.eye(num_classes)[labels]
    return np.mean(np.sum((probs - one_hot) ** 2, axis=1))


def compute_nll(probs, labels):
    """Negative Log-Likelihood."""
    clipped = np.clip(probs[range(len(labels)), labels], 1e-10, 1)
    return -np.mean(np.log(clipped))


def plot_reliability_diagram(probs, labels, model_name="Model",
                              n_bins=15, save_path=None):
    """Publication-quality reliability diagram."""
    ece, bin_data = compute_ece(probs, labels, n_bins)
    brier = compute_brier_score(probs, labels)
    nll = compute_nll(probs, labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Reliability diagram
    confs = [b[0] for b in bin_data if b[2] > 0]
    accs = [b[1] for b in bin_data if b[2] > 0]
    counts = [b[2] for b in bin_data if b[2] > 0]

    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', alpha=0.5)
    ax1.bar(confs, accs, width=1/n_bins*0.8, alpha=0.6,
            color=COLORS[1], edgecolor='black', linewidth=0.5,
            label=f'{model_name}')
    ax1.set_xlabel('Predicted Confidence', fontsize=12)
    ax1.set_ylabel('Observed Accuracy', fontsize=12)
    ax1.set_title(f'Reliability Diagram — {model_name}', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.text(0.05, 0.92, f'ECE = {ece:.4f}\nBrier = {brier:.4f}\nNLL = {nll:.4f}',
             transform=ax1.transAxes, fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Right: confidence histogram
    confidences = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels)
    ax2.hist(confidences[correct], bins=30, alpha=0.6, color=COLORS[2],
             label='Correct', density=True)
    ax2.hist(confidences[~correct], bins=30, alpha=0.6, color=COLORS[0],
             label='Incorrect', density=True)
    ax2.set_xlabel('Prediction Confidence', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.axvline(x=np.median(confidences), color='black', linestyle='--',
                alpha=0.5, label='Median')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

    return {'ece': ece, 'brier': brier, 'nll': nll}


# ════════════════════════════════════════════════════════════════════════
# 3. SELECTIVE PREDICTION (COVERAGE-ACCURACY)
# ════════════════════════════════════════════════════════════════════════

def coverage_accuracy_curve(probs, labels, n_points=100):
    """Compute coverage-accuracy trade-off curve.

    At coverage c: keep the c% most confident predictions, measure accuracy.
    Lower coverage = higher accuracy (easier cases).
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels)

    sorted_idx = np.argsort(-confidences)  # most confident first
    sorted_correct = correct[sorted_idx]

    coverages, accuracies = [], []
    for i in range(1, len(sorted_idx) + 1):
        cov = i / len(sorted_idx)
        acc = sorted_correct[:i].mean()
        coverages.append(cov)
        accuracies.append(acc)

    # Sample n_points evenly
    indices = np.linspace(0, len(coverages) - 1, n_points, dtype=int)
    return np.array(coverages)[indices], np.array(accuracies)[indices]


def compute_aurc(probs, labels):
    """Area Under Risk-Coverage Curve (lower = better).
    Risk = 1 - accuracy at given coverage level.
    """
    coverages, accuracies = coverage_accuracy_curve(probs, labels, n_points=1000)
    risks = 1.0 - accuracies
    aurc = np.trapezoid(risks, coverages)
    return aurc


def find_optimal_thresholds(probs, labels, target_acc=0.98):
    """Find theta_high and theta_low for selective prediction.

    theta_high: minimum confidence to auto-diagnose (target >= target_acc)
    theta_low:  below this → abstain entirely

    CLINICAL CONSTRAINT: θ_high - θ_low >= 0.30
    This ensures a meaningful ASSIST zone (15-25% of cases).
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels)

    # Find theta_high: lowest confidence where accuracy >= target
    best_theta_high = 0.90  # default
    for t in np.arange(0.50, 0.99, 0.01):
        mask = confidences >= t
        if mask.sum() > 0:
            acc = correct[mask].mean()
            cov = mask.mean()
            if acc >= target_acc and cov > 0.3:
                best_theta_high = t
                break

    # theta_low: guarantee meaningful ASSIST zone
    # Use bottom 10th percentile, but ensure gap >= 0.30
    percentile_low = np.percentile(confidences, 10)
    best_theta_low = min(percentile_low, best_theta_high - 0.30)
    best_theta_low = max(best_theta_low, 0.20)  # never below 0.20

    # Final safety: ensure θ_low < θ_high always
    if best_theta_low >= best_theta_high:
        best_theta_low = best_theta_high - 0.30

    return best_theta_high, best_theta_low


def selective_prediction(probs, labels, theta_high, theta_low,
                         class_names=None):
    """Apply 3-tier selective prediction.

    Returns:
        dict with statistics for AUTO/ASSIST/ABSTAIN regions
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels)

    auto_mask = confidences >= theta_high
    abstain_mask = confidences < theta_low
    assist_mask = ~auto_mask & ~abstain_mask

    results = {}
    for name, mask in [('AUTO', auto_mask), ('ASSIST', assist_mask), ('ABSTAIN', abstain_mask)]:
        n = mask.sum()
        acc = correct[mask].mean() if n > 0 else 0
        cov = n / len(labels) * 100
        results[name] = {
            'count': int(n),
            'coverage': cov,
            'accuracy': acc,
            'mean_confidence': confidences[mask].mean() if n > 0 else 0,
        }

        if n > 0 and class_names and name == 'ABSTAIN':
            abstain_classes = Counter(labels[mask])
            results[name]['class_distribution'] = {
                class_names[c]: cnt for c, cnt in abstain_classes.items()
            }

    return results


# ════════════════════════════════════════════════════════════════════════
# 3b. ★ CLINICAL RISK-AWARE SELECTIVE PREDICTION (NOVEL)
# ════════════════════════════════════════════════════════════════════════
# INSIGHT (20+ years clinical AI):
#
# Confidence-only thresholds are CLINICALLY DANGEROUS because they
# treat all errors equally. But in brain tumor diagnosis:
#
#   Missing a GLIOMA (say "benign" when it's malignant) = FATAL
#   Over-detecting a MENINGIOMA (say "malignant" when it's benign) = SAFE
#
# The asymmetric cost of errors MUST be encoded into the decision
# system. This is what separates an AI engineer from a clinical AI
# researcher: understanding that 95% confidence on "no tumor"
# might still warrant ASSIST mode because the cost of being wrong
# is catastrophic.
#
# References:
#   - Kompa et al., "Second opinion needed: communicating uncertainty
#     in medical ML" (npj Digital Medicine, 2021)
#   - Rajpurkar et al., "AI in Health and Medicine" (Nature Med, 2022)
# ════════════════════════════════════════════════════════════════════════

# Clinical misclassification cost matrix
# Row = TRUE class, Col = PREDICTED class
# Higher = more dangerous misclassification
# Scale: 0 (correct) to 10 (catastrophic)
CLINICAL_COST_MATRIX = np.array([
    #  pred: glioma  menin  notumor  pituit
    [  0.0,   3.0,    8.0,    2.0 ],  # true: glioma   (missing glioma → notumor = FATAL)
    [  2.0,   0.0,    5.0,    1.5 ],  # true: meningioma
    [  4.0,   3.0,    0.0,    2.0 ],  # true: notumor  (false alarm = unnecessary surgery)
    [  2.0,   1.5,    5.0,    0.0 ],  # true: pituitary
])

# Per-class risk multiplier: how much EXTRA caution is needed
# when predicting this class. Higher = need more confidence.
#
# "notumor" prediction is HIGH RISK because if we're wrong,
# the patient goes home thinking they're healthy.
# "glioma" prediction is LOWER RISK because if we're wrong,
# the patient still gets further evaluation.
CLASS_RISK_WEIGHTS = {
    'glioma':     0.9,   # Predicting glioma is "safe" — triggers treatment
    'meningioma': 1.0,   # Neutral — mostly benign
    'notumor':    1.4,   # HIGH RISK — patient leaves if we say "no tumor"
    'pituitary':  1.0,   # Neutral
}


def risk_aware_selective_prediction(probs, labels, theta_high, theta_low,
                                     class_names, entropy=None):
    """Clinical risk-aware 3-tier selective prediction.

    Key difference from vanilla selective prediction:
    - theta_high is ADJUSTED per-class based on clinical risk
    - "notumor" prediction requires HIGHER confidence to be AUTO
    - "glioma" prediction can be AUTO at LOWER confidence (safer to over-detect)
    - High entropy cases are escalated regardless of confidence

    Args:
        probs: (N, C) softmax probabilities
        labels: (N,) ground truth labels
        theta_high: base AUTO threshold
        theta_low: base ABSTAIN threshold
        class_names: list of class names
        entropy: (N,) predictive entropy (optional, for entropy-aware gating)

    Returns:
        dict with risk-aware tier assignments and clinical safety metrics
    """
    N = len(labels)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels)

    # ── Risk-adjusted thresholds per class ──
    risk_adjusted_theta = {}
    for i, name in enumerate(class_names):
        weight = CLASS_RISK_WEIGHTS.get(name, 1.0)
        adjusted = min(theta_high * weight, 0.99)
        risk_adjusted_theta[i] = adjusted

    # ── Assign tiers ──
    tiers = np.full(N, 'ASSIST', dtype=object)
    for i in range(N):
        pred_cls = predictions[i]
        conf = confidences[i]
        adjusted_theta = risk_adjusted_theta[pred_cls]

        if conf >= adjusted_theta:
            tiers[i] = 'AUTO'
        elif conf < theta_low:
            tiers[i] = 'ABSTAIN'

        # ENTROPY GATE: high uncertainty → never AUTO, regardless of confidence
        if entropy is not None and entropy[i] > 1.0:
            if tiers[i] == 'AUTO':
                tiers[i] = 'ASSIST'

    # ── Compute stats ──
    results = {'risk_adjusted_thresholds': risk_adjusted_theta}
    for tier_name in ['AUTO', 'ASSIST', 'ABSTAIN']:
        mask = (tiers == tier_name)
        n = mask.sum()
        acc = correct[mask].mean() if n > 0 else 0
        cov = n / N * 100
        results[tier_name] = {
            'count': int(n),
            'coverage': cov,
            'accuracy': acc,
            'mean_confidence': confidences[mask].mean() if n > 0 else 0,
        }

        if n > 0 and tier_name == 'ABSTAIN':
            abstain_classes = Counter(labels[mask])
            results[tier_name]['class_distribution'] = {
                class_names[c]: cnt for c, cnt in abstain_classes.items()
            }

    # ── Clinical Safety Metrics (NOVEL) ──
    # 1. Catch Rate: % of actual errors that land in ASSIST/ABSTAIN
    errors = ~correct
    if errors.sum() > 0:
        errors_caught = ((tiers[errors] == 'ASSIST') | (tiers[errors] == 'ABSTAIN')).sum()
        results['error_catch_rate'] = float(errors_caught / errors.sum())
    else:
        results['error_catch_rate'] = 1.0

    # 2. Critical Miss Rate: % of gliomas misclassified AND in AUTO mode
    glioma_idx = class_names.index('glioma') if 'glioma' in class_names else 0
    glioma_mask = (labels == glioma_idx)
    glioma_wrong_auto = glioma_mask & ~correct & (tiers == 'AUTO')
    if glioma_mask.sum() > 0:
        results['critical_miss_rate'] = float(glioma_wrong_auto.sum() / glioma_mask.sum())
    else:
        results['critical_miss_rate'] = 0.0

    # 3. Expected Clinical Cost
    cost_sum = 0.0
    for i in range(N):
        if tiers[i] == 'AUTO' and not correct[i]:
            cost_sum += CLINICAL_COST_MATRIX[labels[i], predictions[i]]
    results['expected_auto_cost'] = cost_sum / max(1, (tiers == 'AUTO').sum())

    # 4. Safety Score: composite (higher = safer)
    results['safety_score'] = (
        results['error_catch_rate'] * 0.4 +
        (1 - results['critical_miss_rate']) * 0.4 +
        results['AUTO']['accuracy'] * 0.2
    )

    return results, tiers


def plot_selective_prediction(probs, labels, theta_high, theta_low,
                               model_name="Model", save_path=None):
    """Publication-quality selective prediction visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Coverage-Accuracy curve
    covs, accs = coverage_accuracy_curve(probs, labels)
    axes[0].plot(covs * 100, accs * 100, color=COLORS[1], linewidth=2.5)
    axes[0].fill_between(covs * 100, accs * 100, alpha=0.15, color=COLORS[1])
    axes[0].set_xlabel('Coverage (%)', fontsize=12)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Coverage-Accuracy Trade-off', fontsize=14, fontweight='bold')
    aurc = compute_aurc(probs, labels)
    axes[0].text(0.05, 0.05, f'AURC = {aurc:.4f}',
                 transform=axes[0].transAxes, fontsize=12,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Mark operating points
    for c in [90, 95, 100]:
        idx = np.argmin(np.abs(covs - c/100))
        axes[0].scatter(c, accs[idx]*100, s=80, zorder=5, color=COLORS[0])
        axes[0].annotate(f'{accs[idx]*100:.1f}%', (c, accs[idx]*100),
                         textcoords="offset points", xytext=(5, 10), fontsize=10)

    # 2. Three-tier visualization
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels)

    auto_c = confidences[confidences >= theta_high]
    auto_ok = correct[confidences >= theta_high]
    assist_mask = (confidences >= theta_low) & (confidences < theta_high)
    assist_c = confidences[assist_mask]
    assist_ok = correct[assist_mask]
    abstain_c = confidences[confidences < theta_low]

    if len(auto_c) > 0:
        axes[1].hist(auto_c[auto_ok], bins=20, alpha=0.6, color='#2A9D8F', label='AUTO ✓')
        axes[1].hist(auto_c[~auto_ok], bins=20, alpha=0.6, color='#E63946', label='AUTO ✗')
    if len(assist_c) > 0:
        axes[1].hist(assist_c, bins=20, alpha=0.4, color='#E9C46A', label='ASSIST')
    if len(abstain_c) > 0:
        axes[1].hist(abstain_c, bins=20, alpha=0.4, color='#94A3B8', label='ABSTAIN')
    axes[1].axvline(theta_high, color='green', linestyle='--', linewidth=2, label=f'θ_high={theta_high:.2f}')
    axes[1].axvline(theta_low, color='red', linestyle='--', linewidth=2, label=f'θ_low={theta_low:.2f}')
    axes[1].set_xlabel('Prediction Confidence', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Three-Tier Decision Regions', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=9)

    # 3. Summary bar chart
    sp = selective_prediction(probs, labels, theta_high, theta_low, cfg.class_names)
    tiers = ['AUTO', 'ASSIST', 'ABSTAIN']
    tier_colors = ['#2A9D8F', '#E9C46A', '#94A3B8']
    x = np.arange(3)
    coverage = [sp[t]['coverage'] for t in tiers]
    accuracy = [sp[t]['accuracy'] * 100 for t in tiers]

    bar_w = 0.35
    axes[2].bar(x - bar_w/2, coverage, bar_w, color=tier_colors, alpha=0.7, label='Coverage %')
    axes[2].bar(x + bar_w/2, accuracy, bar_w, color=tier_colors, edgecolor='black',
                linewidth=1, label='Accuracy %')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(tiers, fontsize=12)
    axes[2].set_ylabel('Percentage', fontsize=12)
    axes[2].set_title('Selective Prediction Summary', fontsize=14, fontweight='bold')
    axes[2].legend()
    for i, (c, a) in enumerate(zip(coverage, accuracy)):
        axes[2].text(i - bar_w/2, c + 1, f'{c:.1f}%', ha='center', fontsize=9)
        axes[2].text(i + bar_w/2, a + 1, f'{a:.1f}%', ha='center', fontsize=9)

    fig.suptitle(f'{model_name} — Uncertainty-Aware Selective Prediction',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    return sp


# ════════════════════════════════════════════════════════════════════════
# 4. RUN UNCERTAINTY ANALYSIS ON NEUROFUSIONNET
# ════════════════════════════════════════════════════════════════════════

def run_uncertainty_analysis(all_models, test_results, data_dict):
    """Run full uncertainty engine and selective prediction pipeline."""
    nf_val_loader = data_dict['nf_val_loader']
    nf_test_loader = data_dict['nf_test_loader']
    
    print(f"\n{'='*60}")
    print("🔬 Uncertainty Analysis — NeuroFusionNet")
    print(f"{'='*60}")

    nf_model = all_models['NeuroFusionNet']

    # MC Dropout inference
    print(f"\n⏳ MC Dropout (T={cfg.mc_dropout_T}) — this takes ~2-3 minutes...")
    mc_results = mc_dropout_inference(nf_model, nf_test_loader, device, T=cfg.mc_dropout_T)

    mc_acc = accuracy_score(mc_results['labels'], mc_results['preds'])
    mc_f1 = f1_score(mc_results['labels'], mc_results['preds'], average='macro')
    print(f"   MC Dropout Accuracy: {mc_acc:.4f} (vs standard: {test_results['NeuroFusionNet']['accuracy']:.4f})")
    print(f"   MC Dropout F1:       {mc_f1:.4f}")

    # Calibration
    print("\n📊 Calibration Analysis (BEFORE post-hoc calibration):")
    cal_metrics_before = plot_reliability_diagram(
        mc_results['mean_probs'], mc_results['labels'],
        model_name='NeuroFusionNet (Before Calibration)',
        save_path=os.path.join(cfg.output_dir, 'fig_reliability_before_cal.png')
    )
    print(f"   ECE:   {cal_metrics_before['ece']:.4f}")
    print(f"   Brier: {cal_metrics_before['brier']:.4f}")
    print(f"   NLL:   {cal_metrics_before['nll']:.4f}")


    # ════════════════════════════════════════════════════════════════════════
    # 4b. ★ POST-HOC TEMPERATURE SCALING (Guo et al., ICML 2017)
    # ════════════════════════════════════════════════════════════════════════
    # KEY CORRECTION: Temperature Scaling must be learned on a held-out
    # validation set AFTER training, using NLL (not CE with label smoothing).
    # This is fundamentally different from a learnable nn.Parameter trained
    # jointly with the main loss — that approach optimizes for accuracy,
    # NOT calibration.
    #
    # References:
    #   - Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)
    #   - Minderer et al., "Revisiting the Calibration of Modern NNs" (NeurIPS 2021)
    # ════════════════════════════════════════════════════════════════════════

    class PostHocTemperatureScaling:
        """Post-hoc temperature scaling for model calibration.

        Learns a single scalar T on validation logits to minimize NLL.
        After optimization, dividing logits by T produces well-calibrated
        probability estimates — i.e., when the model says 90% confident,
        it is correct ~90% of the time (reliability diagram becomes diagonal).
        """

        def __init__(self):
            self.temperature = 1.0  # Will be optimized

        def fit(self, val_logits, val_labels, lr=0.01, max_iter=200):
            """Learn optimal temperature on validation set.

            Args:
                val_logits: (N, C) raw logits from the model (BEFORE softmax)
                val_labels: (N,) ground truth labels
                lr: learning rate for L-BFGS optimizer
                max_iter: maximum optimization iterations

            Returns:
                optimal temperature (float)
            """
            # Move to device and require gradients
            temperature = nn.Parameter(torch.ones(1, device=val_logits.device) * 1.5)
            optimizer = optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
            criterion = nn.CrossEntropyLoss()

            def closure():
                optimizer.zero_grad()
                loss = criterion(val_logits / temperature, val_labels)
                loss.backward()
                return loss

            optimizer.step(closure)
            self.temperature = temperature.item()

            # Safety: clamp to reasonable range
            self.temperature = max(0.1, min(self.temperature, 10.0))

            return self.temperature

        def calibrate(self, logits):
            """Apply learned temperature to logits."""
            return logits / self.temperature

        def calibrate_probs(self, logits):
            """Apply learned temperature and return calibrated probabilities."""
            return F.softmax(logits / self.temperature, dim=-1)


    @torch.no_grad()
    def collect_val_logits(model, val_loader, device):
        """Collect raw logits from validation set for post-hoc calibration."""
        model.eval()
        if hasattr(model, 'disable_mc_dropout'):
            model.disable_mc_dropout()

        amp_dtype = torch.bfloat16 if cfg.amp_dtype == 'bfloat16' else torch.float16
        all_logits, all_labels = [], []

        for images, labels in val_loader:
            images = images.to(device)
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                output = model(images)
                logits = output['logits'] if isinstance(output, dict) else output
            all_logits.append(logits.float().cpu())
            all_labels.append(labels)

        return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


    # ── Learn optimal temperature on validation set ──
    print("\n🌡️ Post-Hoc Temperature Scaling (Guo et al., ICML 2017):")
    val_logits, val_labels = collect_val_logits(nf_model, nf_val_loader, device)
    print(f"   Collected {len(val_logits)} validation logits")

    temp_scaler = PostHocTemperatureScaling()
    optimal_T = temp_scaler.fit(val_logits, val_labels)
    print(f"   Optimal temperature: T = {optimal_T:.4f}")

    # Apply to model buffer for downstream use
    nf_model.temperature.data.fill_(optimal_T)
    print(f"   ✅ Applied T={optimal_T:.4f} to model.temperature buffer")

    # ── Re-run MC Dropout with calibrated temperature ──
    print("\n⏳ Re-running MC Dropout with calibrated temperature...")
    mc_results_cal = mc_dropout_inference(nf_model, nf_test_loader, device, T=cfg.mc_dropout_T)

    print("\n📊 Calibration Analysis (AFTER post-hoc calibration):")
    cal_metrics_after = plot_reliability_diagram(
        mc_results_cal['mean_probs'], mc_results_cal['labels'],
        model_name='NeuroFusionNet (After Calibration)',
        save_path=os.path.join(cfg.output_dir, 'fig_reliability_after_cal.png')
    )
    print(f"   ECE:   {cal_metrics_before['ece']:.4f} → {cal_metrics_after['ece']:.4f} "
          f"({'↓ IMPROVED' if cal_metrics_after['ece'] < cal_metrics_before['ece'] else '↑ worse'})")
    print(f"   Brier: {cal_metrics_before['brier']:.4f} → {cal_metrics_after['brier']:.4f}")
    print(f"   NLL:   {cal_metrics_before['nll']:.4f} → {cal_metrics_after['nll']:.4f}")

    # Use calibrated results for downstream
    mc_results = mc_results_cal
    cal_metrics = cal_metrics_after

    # Save calibration comparison
    test_results['NeuroFusionNet']['calibration_before'] = cal_metrics_before
    test_results['NeuroFusionNet']['calibration_after'] = cal_metrics_after
    test_results['NeuroFusionNet']['optimal_temperature'] = optimal_T

    # Selective Prediction
    print("\n🎯 Selective Prediction:")
    theta_high, theta_low = find_optimal_thresholds(
        mc_results['mean_probs'], mc_results['labels'], target_acc=0.98
    )

    sp_results = selective_prediction(
        mc_results['mean_probs'], mc_results['labels'],
        theta_high, theta_low, cfg.class_names
    )

    aurc = compute_aurc(mc_results['mean_probs'], mc_results['labels'])
    print(f"   AURC (Area Under Risk-Coverage): {aurc:.5f} (lower is better)")

    _ = plot_selective_prediction(
        mc_results['mean_probs'], mc_results['labels'],
        theta_high, theta_low, model_name='NeuroFusionNet',
        save_path=os.path.join(cfg.output_dir, 'fig_selective_prediction.png')
    )

    # Uncertainty Decomposition Visualization
    print("\n🧩 Uncertainty Decomposition:")
    correct = (mc_results['preds'] == mc_results['labels'])
    ent = mc_results['predictive_entropy']
    mi = mc_results['mutual_information']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(ent[correct], bins=30, alpha=0.6, color=COLORS[2], label='Correct', density=True)
    axes[0].hist(ent[~correct], bins=30, alpha=0.6, color=COLORS[0], label='Incorrect', density=True)
    axes[0].set_xlabel('Predictive Entropy (Total Uncertainty)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Total Uncertainty: Correct vs Incorrect', fontweight='bold')
    axes[0].legend()

    axes[1].hist(mi[correct], bins=30, alpha=0.6, color=COLORS[2], label='Correct', density=True)
    axes[1].hist(mi[~correct], bins=30, alpha=0.6, color=COLORS[0], label='Incorrect', density=True)
    axes[1].set_xlabel('Mutual Information (Epistemic)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Epistemic Uncertainty: Correct vs Incorrect', fontweight='bold')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, 'fig_uncertainty_decomposition.png'))
    plt.show()

    # Save uncertainty results
    test_results['NeuroFusionNet']['uncertainty'] = mc_results
    test_results['NeuroFusionNet']['calibration'] = cal_metrics
    test_results['NeuroFusionNet']['selective'] = sp_results
    test_results['NeuroFusionNet']['aurc'] = aurc
    test_results['NeuroFusionNet']['thresholds'] = (theta_high, theta_low)


    # ════════════════════════════════════════════════════════════════════════
    # 5. ★ CLINICAL RISK-AWARE SELECTIVE PREDICTION
    # ════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*60}")
    print("⚕️ Risk-Aware Selective Prediction (CLINICAL NOVEL)")
    print(f"{'='*60}")

    risk_results, risk_tiers = risk_aware_selective_prediction(
        mc_results['mean_probs'], mc_results['labels'],
        theta_high, theta_low,
        cfg.class_names,
        entropy=mc_results['predictive_entropy']
    )

    # Risk-adjusted thresholds
    print("\n📊 Risk-Adjusted Thresholds (vs vanilla θ_high={:.2f}):".format(theta_high))
    for i, name in enumerate(cfg.class_names):
        weight = CLASS_RISK_WEIGHTS.get(name, 1.0)
        adj_theta = risk_results['risk_adjusted_thresholds'][i]
        direction = "↑ stricter" if weight > 1.0 else ("↓ relaxed" if weight < 1.0 else "= same")
        print(f"   {name:<12}: θ = {adj_theta:.2f} (×{weight}) {direction}")

    # Side-by-side comparison
    print(f"\n{'='*70}")
    print(f"{'':14} │ {'Vanilla SP':^25} │ {'Risk-Aware SP':^25} │")
    print(f"{'─'*70}")
    for tier in ['AUTO', 'ASSIST', 'ABSTAIN']:
        v = sp_results[tier]
        r = risk_results[tier]
        print(f"  {tier:<11} │ {v['count']:>4} ({v['coverage']:>5.1f}%) acc={v['accuracy']:.3f} "
              f"│ {r['count']:>4} ({r['coverage']:>5.1f}%) acc={r['accuracy']:.3f} │")

    # Clinical safety metrics
    print(f"\n{'='*60}")
    print("🏥 Clinical Safety Metrics:")
    print(f"{'='*60}")
    print(f"   Error Catch Rate:    {risk_results['error_catch_rate']:.1%}")
    print(f"     (% of errors caught by ASSIST/ABSTAIN)")
    print(f"   Critical Miss Rate:  {risk_results['critical_miss_rate']:.1%}")
    print(f"     (% gliomas misclassified AND sent as AUTO)")
    print(f"   Expected Auto Cost:  {risk_results['expected_auto_cost']:.3f}")
    print(f"     (average clinical cost of AUTO errors, 0-10 scale)")
    print(f"   Safety Score:        {risk_results['safety_score']:.3f}/1.000")
    print(f"     (composite: catch_rate×0.4 + glioma_safety×0.4 + auto_acc×0.2)")

    # Clinical cost matrix visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: Cost matrix heatmap
    im = axes[0].imshow(CLINICAL_COST_MATRIX, cmap='YlOrRd', vmin=0, vmax=10)
    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels(cfg.class_names, rotation=45, ha='right')
    axes[0].set_yticks(range(4))
    axes[0].set_yticklabels(cfg.class_names)
    axes[0].set_xlabel('Predicted Class', fontsize=12)
    axes[0].set_ylabel('True Class', fontsize=12)
    axes[0].set_title('Clinical Misclassification Cost Matrix', fontsize=13, fontweight='bold')
    for i in range(4):
        for j in range(4):
            color = 'white' if CLINICAL_COST_MATRIX[i, j] > 4 else 'black'
            axes[0].text(j, i, f'{CLINICAL_COST_MATRIX[i, j]:.0f}',
                         ha='center', va='center', fontsize=14, fontweight='bold', color=color)
    plt.colorbar(im, ax=axes[0], label='Cost (0=correct, 10=fatal)')

    # Right: Vanilla vs Risk-Aware comparison
    tiers_list = ['AUTO', 'ASSIST', 'ABSTAIN']
    x = np.arange(3)
    w = 0.3
    vanilla_acc = [sp_results[t]['accuracy']*100 for t in tiers_list]
    risk_acc = [risk_results[t]['accuracy']*100 for t in tiers_list]
    vanilla_cov = [sp_results[t]['coverage'] for t in tiers_list]
    risk_cov = [risk_results[t]['coverage'] for t in tiers_list]

    bars1 = axes[1].bar(x - w/2, vanilla_acc, w, color='#E9C46A', alpha=0.8,
                        label='Vanilla Accuracy', edgecolor='black', linewidth=0.5)
    bars2 = axes[1].bar(x + w/2, risk_acc, w, color='#2A9D8F', alpha=0.8,
                        label='Risk-Aware Accuracy', edgecolor='black', linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tiers_list, fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Vanilla vs Risk-Aware Selective Prediction', fontsize=13, fontweight='bold')
    axes[1].legend()
    for bar, val in zip(bars1, vanilla_acc):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.1f}',
                     ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, risk_acc):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.1f}',
                     ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, 'fig_risk_aware_prediction.png'))
    plt.show()

    # Save results
    test_results['NeuroFusionNet']['risk_aware'] = risk_results
    test_results['NeuroFusionNet']['risk_tiers'] = risk_tiers

    print(f"\n✅ Uncertainty + Risk-Aware analysis complete!")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    data_dict = setup_data()
    all_models, test_results, data_dict = load_all_checkpoints(data_dict)
    run_uncertainty_analysis(all_models, test_results, data_dict)
