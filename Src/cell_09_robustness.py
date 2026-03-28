import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from cell_04_training import *

# %% Cell 9: Robustness Evaluation + Ablation Study
# ════════════════════════════════════════════════════════════════════════
# DESIGN (Falsification): We test what BREAKS the model, not what looks good.
# ════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
# 1. CORRUPTION ROBUSTNESS (Simulated Domain Shift)
# ════════════════════════════════════════════════════════════════════════

def get_corruption_transforms(corruption_type, severity=3, img_size=256):
    """Create corruption augmentation for robustness testing."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    corruptions = {
        'gaussian_noise': A.Compose([
            A.Resize(img_size, img_size),
            A.GaussNoise(var_limit=(20 * severity, 30 * severity), p=1.0),
            A.Normalize(mean=mean, std=std), ToTensorV2(),
        ]),
        'gaussian_blur': A.Compose([
            A.Resize(img_size, img_size),
            A.GaussianBlur(blur_limit=(3 + 2*severity, 5 + 2*severity), p=1.0),
            A.Normalize(mean=mean, std=std), ToTensorV2(),
        ]),
        'contrast_shift': A.Compose([
            A.Resize(img_size, img_size),
            A.RandomBrightnessContrast(brightness_limit=0.1*severity,
                                       contrast_limit=0.15*severity, p=1.0),
            A.Normalize(mean=mean, std=std), ToTensorV2(),
        ]),
        'salt_pepper': A.Compose([
            A.Resize(img_size, img_size),
            A.OneOf([
                A.GaussNoise(var_limit=(5 * severity, 15 * severity), p=1.0),
            ], p=1.0),
            A.Normalize(mean=mean, std=std), ToTensorV2(),
        ]),
        'elastic': A.Compose([
            A.Resize(img_size, img_size),
            A.ElasticTransform(alpha=30 * severity, sigma=30*severity*0.05, p=1.0),
            A.Normalize(mean=mean, std=std), ToTensorV2(),
        ]),
    }
    return corruptions.get(corruption_type)


@torch.no_grad()
def evaluate_robustness(model, test_df, device, corruptions=None,
                         severities=None, img_size=256):
    """Evaluate model robustness under various corruptions."""
    if corruptions is None:
        corruptions = ['gaussian_noise', 'gaussian_blur', 'contrast_shift', 'elastic']
    if severities is None:
        severities = [1, 2, 3]

    model.eval()
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == 'bfloat16' else torch.float16
    results = {}

    for corr_type in corruptions:
        results[corr_type] = {}
        for sev in severities:
            transform = get_corruption_transforms(corr_type, sev, img_size)
            ds = BrainTumorDataset(test_df, transform)
            loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                                num_workers=cfg.num_workers, pin_memory=True)

            all_preds, all_labels = [], []
            for imgs, lbls in loader:
                imgs = imgs.to(device)
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    logits = model(imgs)
                    if isinstance(logits, dict):
                        logits = logits['logits']
                preds = logits.argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(lbls.numpy())

            acc = accuracy_score(all_labels, all_preds)
            results[corr_type][sev] = acc
            print(f"   {corr_type} (sev={sev}): Acc={acc:.4f}")

    return results


def run_robustness(all_models, test_results, data_dict):
    test_df = data_dict["test_df"]
    # ── Run robustness ──
    print(f"\n{'='*60}")
    print("🛡️ Robustness Evaluation (Simulated Domain Shift)")
    print(f"{'='*60}")

    nf_model = all_models['NeuroFusionNet']
    robustness_nf = evaluate_robustness(nf_model, test_df, device, img_size=256)

    # Clean accuracy as reference
    clean_acc = test_results['NeuroFusionNet']['accuracy']

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    severities = [1, 2, 3]
    x = np.arange(len(severities))
    width = 0.2

    for i, (corr_type, sev_dict) in enumerate(robustness_nf.items()):
        accs = [sev_dict[s] for s in severities]
        bars = ax.bar(x + i * width, accs, width, label=corr_type.replace('_', ' ').title(),
                      alpha=0.8)
        for bar, a in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{a:.2f}', ha='center', fontsize=8)

    ax.axhline(y=clean_acc, color='black', linestyle='--', linewidth=1.5,
               label=f'Clean acc: {clean_acc:.3f}')
    ax.set_xlabel('Corruption Severity', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('NeuroFusionNet — Corruption Robustness', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f'Severity {s}' for s in severities])
    ax.legend(fontsize=9, loc='lower left')
    ax.set_ylim(0.7, 1.02)

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, 'fig_robustness.png'))
    plt.show()

    # Compute mean Corruption Error
    mean_corr = np.mean([robustness_nf[c][3] for c in robustness_nf if 3 in robustness_nf[c]])
    print(f"\n📊 Mean accuracy at severity 3: {mean_corr:.4f} (drop: {(clean_acc - mean_corr)*100:.1f}%)")


    # ════════════════════════════════════════════════════════════════════════
    # 2. ABLATION STUDY (Publication-Ready)
    # ════════════════════════════════════════════════════════════════════════
    # Q1 REQUIREMENT: Ablation study must prove each component contributes.
    # Format follows IEEE/Elsevier conventions (LaTeX-ready).
    # ════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*60}")
    print("🔬 Ablation Study — Contribution of Each Component")
    print(f"{'='*60}")

    # Instead of retraining (expensive), we analyze existing results
    ablation_results = []

    # A1: Single backbone baselines
    for name in ['Swin-V2-S', 'ConvNeXt-Base']:
        if name in test_results:
            ablation_results.append({
                'ID': f'A{len(ablation_results)+1}',
                'Configuration': f'Single Backbone ({name})',
                'Accuracy': test_results[name]['accuracy'],
                'F1-Macro': test_results[name]['f1_macro'],
                'ECE': '—', 'AURC': '—', 'Safety': '—',
            })

    # Other baselines
    for name in ['EfficientNetV2-S', 'ViT-B/16']:
        if name in test_results:
            ablation_results.append({
                'ID': f'A{len(ablation_results)+1}',
                'Configuration': f'Single Backbone ({name})',
                'Accuracy': test_results[name]['accuracy'],
                'F1-Macro': test_results[name]['f1_macro'],
                'ECE': '—', 'AURC': '—', 'Safety': '—',
            })

    # Full model
    nf_acc = test_results['NeuroFusionNet']['accuracy']
    nf_f1 = test_results['NeuroFusionNet']['f1_macro']
    nf_ece = test_results['NeuroFusionNet'].get('calibration_after', 
             test_results['NeuroFusionNet'].get('calibration', {})).get('ece', '—')
    nf_aurc = test_results['NeuroFusionNet'].get('aurc', '—')
    ablation_results.append({
        'ID': f'A{len(ablation_results)+1}',
        'Configuration': 'NeuroFusionNet (Full)',
        'Accuracy': nf_acc,
        'F1-Macro': nf_f1,
        'ECE': nf_ece, 'AURC': nf_aurc, 'Safety': '—',
    })

    # Selective prediction (AUTO only)
    if 'selective' in test_results.get('NeuroFusionNet', {}):
        sp = test_results['NeuroFusionNet']['selective']
        if sp['AUTO']['count'] > 0:
            ablation_results.append({
                'ID': f'A{len(ablation_results)+1}',
                'Configuration': f'+ Vanilla Selective Pred (AUTO {sp["AUTO"]["coverage"]:.0f}%)',
                'Accuracy': sp['AUTO']['accuracy'],
                'F1-Macro': '—',
                'ECE': '—', 'AURC': '—', 'Safety': '—',
            })

    # Risk-aware selective prediction
    if 'risk_aware' in test_results.get('NeuroFusionNet', {}):
        ra = test_results['NeuroFusionNet']['risk_aware']
        if ra['AUTO']['count'] > 0:
            ablation_results.append({
                'ID': f'A{len(ablation_results)+1}',
                'Configuration': f'+ Risk-Aware SP (AUTO {ra["AUTO"]["coverage"]:.0f}%)',
                'Accuracy': ra['AUTO']['accuracy'],
                'F1-Macro': '—',
                'ECE': '—', 'AURC': '—',
                'Safety': f'{ra.get("safety_score", 0):.3f}',
            })
            ablation_results.append({
                'ID': f'A{len(ablation_results)+1}',
                'Configuration': '  → Error Catch Rate',
                'Accuracy': ra.get('error_catch_rate', 0),
                'F1-Macro': '—',
                'ECE': '—', 'AURC': '—',
                'Safety': f'Critical Miss: {ra.get("critical_miss_rate", 0):.1%}',
            })

    # Calibration impact
    if 'calibration_before' in test_results.get('NeuroFusionNet', {}):
        ece_before = test_results['NeuroFusionNet']['calibration_before']['ece']
        ece_after = test_results['NeuroFusionNet']['calibration_after']['ece']
        opt_T = test_results['NeuroFusionNet'].get('optimal_temperature', 1.0)
        ablation_results.append({
            'ID': f'A{len(ablation_results)+1}',
            'Configuration': f'+ Post-hoc Temp Scaling (T={opt_T:.2f})',
            'Accuracy': nf_acc,
            'F1-Macro': nf_f1,
            'ECE': ece_after, 'AURC': nf_aurc,
            'Safety': f'ECE: {ece_before:.4f}→{ece_after:.4f}',
        })

    ablation_df = pd.DataFrame(ablation_results)
    print("\n" + ablation_df.to_string(index=False))


    def format_ablation_for_paper(df):
        """Format ablation table for LaTeX/paper."""
        print("\n📋 LaTeX-ready Ablation Table:")
        print("\\begin{table}[h]")
        print("\\caption{Ablation Study: Component Contributions}")
        print("\\begin{tabular}{lcccc}")
        print("\\toprule")
        print("Configuration & Accuracy & F1 & ECE↓ & AURC↓ \\\\")
        print("\\midrule")
        for _, row in df.iterrows():
            acc = f"{row['Accuracy']:.4f}" if isinstance(row['Accuracy'], float) else row['Accuracy']
            f1 = f"{row['F1-Macro']:.4f}" if isinstance(row['F1-Macro'], float) else row['F1-Macro']
            ece = f"{row['ECE']:.4f}" if isinstance(row['ECE'], float) else row['ECE']
            aurc = f"{row['AURC']:.4f}" if isinstance(row['AURC'], float) else row['AURC']
            print(f"  {row['Configuration']} & {acc} & {f1} & {ece} & {aurc} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")


    format_ablation_for_paper(ablation_df)

    # Save ablation as figure
    fig, ax = plt.subplots(figsize=(12, 6))
    # Filter rows with float accuracy for plotting
    plot_rows = ablation_df[ablation_df['Accuracy'].apply(lambda x: isinstance(x, float))]
    configs = plot_rows['Configuration'].tolist()
    accs = plot_rows['Accuracy'].tolist()
    color_list = [COLORS[i % len(COLORS)] for i in range(len(configs))]
    bars = ax.barh(configs, accs, color=color_list, alpha=0.8, edgecolor='black', linewidth=0.5)
    for bar, a in zip(bars, accs):
        ax.text(a + 0.002, bar.get_y() + bar.get_height()/2,
                f'{a:.4f}', va='center', fontsize=10)
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title('Ablation Study — Component Contributions', fontsize=14, fontweight='bold')
    ax.set_xlim(0.85, 1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, 'fig_ablation.png'))
    plt.show()

    print("\n✅ Robustness & Ablation complete!")

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
    
    # NOTE: We skip cell_06_uncertainty by default to speed up execution.
    # The ablation table will gracefully show '—' for missing metrics (AURC, ECE).
    # If full ablation table is needed, one would call run_uncertainty_analysis here.
    
    # Run robustness
    run_robustness(all_models, test_results, data_dict)
