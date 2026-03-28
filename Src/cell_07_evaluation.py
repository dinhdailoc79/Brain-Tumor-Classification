import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from cell_04_training import *

# %% Cell 7: Evaluation — Confusion Matrix, Per-Class, Comparison
# ════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(labels, preds, class_names, title="Confusion Matrix",
                          save_path=None):
    """Publication-quality confusion matrix."""
    cm = confusion_matrix(labels, preds)
    cm_pct = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Absolute
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor='gray')
    axes[0].set_title(f'{title} (Counts)', fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    # Percentage
    sns.heatmap(cm_pct, annot=True, fmt='.2%', cmap='Reds', ax=axes[1],
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor='gray')
    axes[1].set_title(f'{title} (Normalized)', fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    # Highlight glioma↔meningioma
    for ax in axes:
        for ci, cj in [(0, 1), (1, 0)]:
            ax.add_patch(plt.Rectangle((cj, ci), 1, 1, fill=False,
                                        edgecolor='red', linewidth=2.5))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

    gm_confusion = cm[0, 1] + cm[1, 0]
    return cm, gm_confusion


def run_evaluation(test_results, all_histories=None):
    if all_histories is None:
        all_histories = {}

    # ── All models comparison ──
    print(f"\n{'='*60}")
    print("📊 Full Evaluation — All Models")
    print(f"{'='*60}")

    comparison_data = []
    for name in list(cfg.baseline_models.keys()) + ['NeuroFusionNet']:
        r = test_results[name]
        labels, preds, probs = r['labels'], r['preds'], r['probs']

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')
        prec = precision_score(labels, preds, average='macro')
        rec = recall_score(labels, preds, average='macro')
        kappa = cohen_kappa_score(labels, preds)
        mcc = matthews_corrcoef(labels, preds)

        cm = confusion_matrix(labels, preds)
        gm_conf = cm[0, 1] + cm[1, 0]
        glioma_recall = cm[0, 0] / max(cm[0].sum(), 1)
        menin_recall = cm[1, 1] / max(cm[1].sum(), 1)
        notumor_fp = 1 - (cm[2, 2] / max(cm[2].sum(), 1))

        comparison_data.append({
            'Model': name,
            'Accuracy': acc, 'F1-Macro': f1,
            'Precision': prec, 'Recall': rec,
            'Kappa': kappa, 'MCC': mcc,
            'GM-Confusion': gm_conf,
            'Glioma-Recall': glioma_recall,
            'Menin-Recall': menin_recall,
            'NoTumor-FP': notumor_fp,
        })

    comp_df = pd.DataFrame(comparison_data)
    print("\n" + comp_df.to_string(index=False))


    # ════════════════════════════════════════════════════════════════════════
    # STATISTICAL RIGOR: Confidence Intervals + McNemar's Test
    # ════════════════════════════════════════════════════════════════════════
    # Q1 REQUIREMENT: Single-run metrics are anecdotal evidence.
    # Bootstrap CIs provide interval estimates; McNemar's test provides
    # pairwise significance between models.
    #
    # References:
    #   - Efron & Tibshirani, "An Introduction to the Bootstrap" (1993)
    #   - Dietterich, "Approximate Statistical Tests for Comparing
    #     Supervised Classification Algorithms" (Neural Computation, 1998)
    # ════════════════════════════════════════════════════════════════════════

    def compute_bootstrap_ci(labels, preds, metric_fn, n_bootstrap=1000,
                              ci_level=0.95, seed=42):
        """Compute bootstrap confidence interval for a metric.

        Args:
            labels: ground truth labels
            preds: model predictions
            metric_fn: function(labels, preds) → float
            n_bootstrap: number of bootstrap samples
            ci_level: confidence level (0.95 = 95% CI)

        Returns:
            (point_estimate, lower_bound, upper_bound)
        """
        rng = np.random.RandomState(seed)
        n = len(labels)
        estimates = []

        for _ in range(n_bootstrap):
            idx = rng.randint(0, n, size=n)
            est = metric_fn(labels[idx], preds[idx])
            estimates.append(est)

        estimates = np.array(estimates)
        alpha = 1 - ci_level
        lower = np.percentile(estimates, alpha / 2 * 100)
        upper = np.percentile(estimates, (1 - alpha / 2) * 100)
        point = metric_fn(labels, preds)

        return point, lower, upper


    def mcnemar_test(labels, preds_a, preds_b, model_a_name="A", model_b_name="B"):
        """McNemar's test for comparing two classifiers.

        Tests whether the disagreement between two models is symmetric.
        If p < 0.05, the models are significantly different.

        Returns:
            dict with statistic, p_value, and interpretation
        """
        correct_a = (preds_a == labels)
        correct_b = (preds_b == labels)

        # b: A correct, B wrong | c: A wrong, B correct
        b = np.sum(correct_a & ~correct_b)
        c = np.sum(~correct_a & correct_b)

        # McNemar's test with continuity correction
        if b + c == 0:
            return {'statistic': 0, 'p_value': 1.0,
                    'interpretation': 'No disagreement between models'}

        from scipy.stats import chi2
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - chi2.cdf(statistic, df=1)

        if p_value < 0.001:
            sig = "★★★ Highly significant (p<0.001)"
        elif p_value < 0.01:
            sig = "★★ Significant (p<0.01)"
        elif p_value < 0.05:
            sig = "★ Significant (p<0.05)"
        else:
            sig = "Not significant (p≥0.05)"

        return {
            'model_a': model_a_name, 'model_b': model_b_name,
            'b': int(b), 'c': int(c),
            'statistic': float(statistic), 'p_value': float(p_value),
            'interpretation': sig,
        }


    # ── Confidence Intervals ──
    print(f"\n{'='*75}")
    print("📊 Confidence Intervals (95% Bootstrap, N=1000):")
    print(f"{'='*75}")
    print(f"{'Model':<25} {'Accuracy (95% CI)':<28} {'F1-Macro (95% CI)':<28}")
    print(f"{'─'*75}")

    for name in list(cfg.baseline_models.keys()) + ['NeuroFusionNet']:
        r = test_results[name]
        labels, preds = r['labels'], r['preds']

        acc_pt, acc_lo, acc_hi = compute_bootstrap_ci(
            labels, preds, accuracy_score)
        f1_pt, f1_lo, f1_hi = compute_bootstrap_ci(
            labels, preds, lambda l, p: f1_score(l, p, average='macro', zero_division=0))

        print(f"{name:<25} {acc_pt:.4f} [{acc_lo:.4f}, {acc_hi:.4f}]  "
              f"{f1_pt:.4f} [{f1_lo:.4f}, {f1_hi:.4f}]")

    # ── McNemar's Tests (NeuroFusionNet vs each baseline) ──
    print(f"\n{'='*70}")
    print("📊 McNemar's Test: NeuroFusionNet vs Baselines")
    print(f"{'='*70}")

    nf_preds = test_results['NeuroFusionNet']['preds']
    nf_labels = test_results['NeuroFusionNet']['labels']

    for name in cfg.baseline_models.keys():
        if name in test_results:
            baseline_preds = test_results[name]['preds']
            result = mcnemar_test(nf_labels, nf_preds, baseline_preds,
                                   "NeuroFusionNet", name)
            print(f"  vs {name:<20}: p={result['p_value']:.4f} "
                  f"(b={result['b']}, c={result['c']}) → {result['interpretation']}")

    # ── Confusion matrices ──
    for name in ['NeuroFusionNet']:
        r = test_results[name]
        cm, gm = plot_confusion_matrix(
            r['labels'], r['preds'], cfg.class_names,
            title=f'{name}',
            save_path=os.path.join(cfg.output_dir, f'fig_cm_{name}.png')
        )
        print(f"\n{name}: Glioma↔Meningioma confusion = {gm} cases")

    # ── Training curves ──
    if all_histories:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        for name, history in all_histories.items():
            n = len(history['val_acc'])
            axes[0, 0].plot(range(1, n+1), history['val_acc'], label=name, linewidth=1.5)
            axes[0, 1].plot(range(1, n+1), history['val_f1'], label=name, linewidth=1.5)
            axes[1, 0].plot(range(1, n+1), history['train_loss'], label=name, linewidth=1.5)
            axes[1, 1].plot(range(1, n+1), history['val_loss'], label=name, linewidth=1.5)

        for i, (ax, title) in enumerate(zip(axes.flat,
            ['Validation Accuracy', 'Validation F1-Macro', 'Training Loss', 'Validation Loss'])):
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(cfg.output_dir, 'fig_training_curves.png'))
        plt.show()

    # ── Comparison bar chart ──
    fig, ax = plt.subplots(figsize=(12, 6))
    models = comp_df['Model']
    x = np.arange(len(models))
    w = 0.25
    ax.bar(x - w, comp_df['Accuracy'], w, label='Accuracy', color=COLORS[0], alpha=0.8)
    ax.bar(x, comp_df['F1-Macro'], w, label='F1-Macro', color=COLORS[1], alpha=0.8)
    ax.bar(x + w, comp_df['Recall'], w, label='Recall', color=COLORS[2], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison — Key Metrics', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_ylim(0.85, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, 'fig_model_comparison.png'))
    plt.show()

    print("\n✅ Evaluation complete!")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    from cell_01_setup import *
    from cell_02_data import *
    from cell_03_models import *
    from cell_04_training import *
    from cell_00_load_checkpoints import load_all_checkpoints
    
    # Needs to init setup_data
    data_dict = setup_data()
    # Load model and checkpoints
    all_models, test_results, data_dict = load_all_checkpoints(data_dict)
    
    # Run evaluation directly, skipping cell 06 uncertainty!
    run_evaluation(test_results, all_histories={})
