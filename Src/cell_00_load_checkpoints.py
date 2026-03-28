import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from cell_04_training import *

# %% Cell 0: ★ LOAD TRAINED MODELS (Skip Training)
# ════════════════════════════════════════════════════════════════════════
# RUN THIS INSTEAD OF CELL 5 to skip training.
# Prerequisites:
#   1. Run cells 1-4 first (setup, data, model definitions, trainer)
#   2. Have checkpoint files in cfg.checkpoint_dir:
#      - NeuroFusionNet_best.pth
#      - EfficientNetV2-S_best.pth (or other baselines)
#   3. Then run cells 6-11 as normal
#
# USAGE:
#   Cell 1 → Cell 2 → Cell 3 → Cell 4 → ★ THIS CELL ★ → Cell 6-11
# ════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════
# 1. LOAD BASELINES
# ════════════════════════════════════════════════════════════════════════

def load_all_checkpoints(data_dict, load_baselines=True):
    """Load baseline models and NeuroFusionNet from checkpoints."""
    test_df = data_dict['test_df']
    test_loader = data_dict['test_loader']
    train_df_final = data_dict['train_df_final']
    val_df = data_dict['val_df']
    
    all_models = {}
    test_results = {}
    
    print(f"\n{'='*60}")
    print("💾 Loading pre-trained checkpoints (NO training needed)")
    print(f"{'='*60}")

    # ── Check checkpoint directory ──
    if not os.path.exists(cfg.checkpoint_dir):
        raise FileNotFoundError(
            f"❌ Checkpoint directory not found: {cfg.checkpoint_dir}\n"
            f"   Please upload your checkpoints or run Cell 5 to train."
        )

    ckpt_files = [f for f in os.listdir(cfg.checkpoint_dir) if f.endswith('.pth')]
    print(f"📂 Found {len(ckpt_files)} checkpoints in {cfg.checkpoint_dir}:")
    for f in ckpt_files:
        size_mb = os.path.getsize(os.path.join(cfg.checkpoint_dir, f)) / 1e6
        print(f"   📦 {f} ({size_mb:.1f} MB)")

    for display_name, model_key in cfg.baseline_models.items():
        if not load_baselines:
            continue
        ckpt_name = f'{display_name.replace("/", "_")}_best.pth'
        ckpt_path = os.path.join(cfg.checkpoint_dir, ckpt_name)

        if not os.path.exists(ckpt_path):
            print(f"   ⚠️ {display_name}: checkpoint not found, skipping")
            continue

        print(f"\n🔄 Loading: {display_name}...")
        model = create_baseline(model_key, cfg.num_classes)
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        # Create test loader with correct image size
        img_size = get_model_img_size(model_key)
        if img_size != cfg.img_size:
            _test_ds = BrainTumorDataset(test_df, get_transforms('test', img_size))
            _test_loader = DataLoader(_test_ds, batch_size=cfg.batch_size, shuffle=False,
                                       num_workers=cfg.num_workers, pin_memory=True)
        else:
            _test_loader = test_loader

        # Evaluate
        labels, preds, probs = evaluate_model(model, _test_loader, device)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')
        print(f"   ✅ {display_name}: Acc={acc:.4f} F1={f1:.4f}")

        all_models[display_name] = model
        test_results[display_name] = {
            'labels': labels, 'preds': preds, 'probs': probs,
            'accuracy': acc, 'f1_macro': f1,
        }
    # ════════════════════════════════════════════════════════════════════════
    # 2. LOAD NEUROFUSIONNET
    # ════════════════════════════════════════════════════════════════════════

    nf_ckpt = os.path.join(cfg.checkpoint_dir, 'NeuroFusionNet_best.pth')
    if os.path.exists(nf_ckpt):
        print(f"\n🧠 Loading NeuroFusionNet...")
        neurofusion = NeuroFusionNet(
            num_classes=cfg.num_classes, pretrained=False,  # No need to download pretrained
            hidden_dim=cfg.fusion_dim, dropout=cfg.dropout
        )
        state_dict = torch.load(nf_ckpt, map_location=device, weights_only=True)
        neurofusion.load_state_dict(state_dict)
        neurofusion = neurofusion.to(device)
        neurofusion.eval()

        total_params = sum(p.numel() for p in neurofusion.parameters())
        print(f"   Parameters: {total_params/1e6:.1f}M")

        # Create NeuroFusionNet datasets/loaders
        nf_train_ds = BrainTumorDataset(train_df_final, get_transforms('train', 256))
        nf_val_ds = BrainTumorDataset(val_df, get_transforms('val', 256))
        nf_test_ds = BrainTumorDataset(test_df, get_transforms('test', 256))
        nf_test_loader = DataLoader(nf_test_ds, batch_size=cfg.batch_size, shuffle=False,
                                    num_workers=cfg.num_workers, pin_memory=True)
        nf_val_loader = DataLoader(nf_val_ds, batch_size=cfg.batch_size, shuffle=False,
                                   num_workers=cfg.num_workers, pin_memory=True)

        data_dict['nf_val_loader'] = nf_val_loader
        data_dict['nf_test_loader'] = nf_test_loader

        # Evaluate
        labels, preds, probs = evaluate_model(neurofusion, nf_test_loader, device)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')
        print(f"   ✅ NeuroFusionNet: Acc={acc:.4f} F1={f1:.4f}")

        all_models['NeuroFusionNet'] = neurofusion
        test_results['NeuroFusionNet'] = {
            'labels': labels, 'preds': preds, 'probs': probs,
            'accuracy': acc, 'f1_macro': f1,
        }
    else:
        print(f"❌ NeuroFusionNet checkpoint not found: {nf_ckpt}")

    # ════════════════════════════════════════════════════════════════════════
    # 3. SUMMARY
    # ════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*75}")
    print(f"{'Model':<25} {'Accuracy':<12} {'F1-Macro':<12} {'Params (M)':<12}")
    print(f"{'─'*75}")
    for name in all_models:
        r = test_results[name]
        n_params = sum(p.numel() for p in all_models[name].parameters()) / 1e6
        print(f"{name:<25} {r['accuracy']:<12.4f} {r['f1_macro']:<12.4f} {n_params:<12.1f}")
    print(f"{'='*75}")
    print(f"\n✅ All models loaded! Now run cells 6-11 for analysis + demo.")
    
    return all_models, test_results, data_dict

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    data_dict = setup_data()
    all_models, test_results, data_dict = load_all_checkpoints(data_dict)
