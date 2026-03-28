import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from cell_04_training import *

# %% Cell 5: Train All Models
# ════════════════════════════════════════════════════════════════════════
# Trains 4 baselines + NeuroFusionNet sequentially on A100
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ════════════════════════════════════════════════════════════════════════
    # 1. TRAIN BASELINES
    # ════════════════════════════════════════════════════════════════════════

    for display_name, model_key in cfg.baseline_models.items():
        print(f"\n{'='*60}")
        print(f"🚀 Training: {display_name} ({model_key})")
        print(f"{'='*60}")

        seed_everything(cfg.seed)
        img_size = get_model_img_size(model_key)

        # Create model-specific dataloaders if image size differs
        if img_size != cfg.img_size:
            _train_ds = BrainTumorDataset(train_df_final, get_transforms('train', img_size))
            _val_ds = BrainTumorDataset(val_df, get_transforms('val', img_size))
            _test_ds = BrainTumorDataset(test_df, get_transforms('test', img_size))
            _sampler = get_balanced_sampler(train_df_final['label_idx'].tolist())
            _train_loader = DataLoader(_train_ds, batch_size=cfg.batch_size, sampler=_sampler,
                                        num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
            _val_loader = DataLoader(_val_ds, batch_size=cfg.batch_size, shuffle=False,
                                      num_workers=cfg.num_workers, pin_memory=True)
            _test_loader = DataLoader(_test_ds, batch_size=cfg.batch_size, shuffle=False,
                                       num_workers=cfg.num_workers, pin_memory=True)
        else:
            _train_loader, _val_loader, _test_loader = train_loader, val_loader, test_loader

        model = create_baseline(model_key, cfg.num_classes)
        trainer = BaselineTrainer(model, cfg, device)
        history = trainer.fit(_train_loader, _val_loader)

        # Save checkpoint
        ckpt = os.path.join(cfg.checkpoint_dir, f'{display_name.replace("/", "_")}_best.pth')
        torch.save(trainer.model.state_dict(), ckpt)

        # Evaluate on test
        labels, preds, probs = evaluate_model(trainer.model, _test_loader, device)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')
        print(f"\n📊 {display_name} Test: Acc={acc:.4f} F1={f1:.4f}")
        print(classification_report(labels, preds, target_names=cfg.class_names, digits=4))

        # Store results
        all_models[display_name] = trainer.model
        all_histories[display_name] = history
        test_results[display_name] = {
            'labels': labels, 'preds': preds, 'probs': probs,
            'accuracy': acc, 'f1_macro': f1,
        }

        del trainer; gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("✅ All baselines trained!")
    for name in cfg.baseline_models:
        print(f"   {name}: Acc={test_results[name]['accuracy']:.4f} F1={test_results[name]['f1_macro']:.4f}")

    # ════════════════════════════════════════════════════════════════════════
    # 2. TRAIN NEUROFUSIONNET
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("🧠 Training NeuroFusionNet (Dual-Backbone + Multi-Task Expert Heads)")
    print(f"{'='*60}")

    seed_everything(cfg.seed)

    neurofusion = NeuroFusionNet(
        num_classes=cfg.num_classes, pretrained=True,
        hidden_dim=cfg.fusion_dim, dropout=cfg.dropout
    )
    total_params = sum(p.numel() for p in neurofusion.parameters())
    print(f"   Parameters: {total_params/1e6:.1f}M")

    # NeuroFusionNet uses 256x256 for Swin-V2
    nf_train_ds = BrainTumorDataset(train_df_final, get_transforms('train', 256))
    nf_val_ds = BrainTumorDataset(val_df, get_transforms('val', 256))
    nf_test_ds = BrainTumorDataset(test_df, get_transforms('test', 256))
    nf_sampler = get_balanced_sampler(train_df_final['label_idx'].tolist())

    nf_train_loader = DataLoader(nf_train_ds, batch_size=cfg.batch_size, sampler=nf_sampler,
                                  num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    nf_val_loader = DataLoader(nf_val_ds, batch_size=cfg.batch_size, shuffle=False,
                                num_workers=cfg.num_workers, pin_memory=True)
    nf_test_loader = DataLoader(nf_test_ds, batch_size=cfg.batch_size, shuffle=False,
                                 num_workers=cfg.num_workers, pin_memory=True)

    class_counts_train = Counter(train_df_final['label_idx'].tolist())
    nf_trainer = NeuroFusionTrainer(neurofusion, cfg, device, class_counts=class_counts_train)
    nf_history = nf_trainer.fit(nf_train_loader, nf_val_loader)

    # Save
    ckpt = os.path.join(cfg.checkpoint_dir, 'NeuroFusionNet_best.pth')
    torch.save(nf_trainer.model.state_dict(), ckpt)
    print(f"💾 Saved: {ckpt}")

    # Evaluate
    labels, preds, probs = evaluate_model(nf_trainer.model, nf_test_loader, device)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    print(f"\n📊 NeuroFusionNet Test: Acc={acc:.4f} F1={f1:.4f}")
    print(classification_report(labels, preds, target_names=cfg.class_names, digits=4))

    # Confusion analysis
    cm = confusion_matrix(labels, preds)
    gm_confusion = cm[0, 1] + cm[1, 0]
    print(f"🎯 Glioma↔Meningioma confusion: {gm_confusion} cases")
    print(f"   Glioma recall: {cm[0,0]/cm[0].sum()*100:.1f}%")
    print(f"   Meningioma recall: {cm[1,1]/cm[1].sum()*100:.1f}%")

    all_models['NeuroFusionNet'] = nf_trainer.model
    all_histories['NeuroFusionNet'] = nf_history
    test_results['NeuroFusionNet'] = {
        'labels': labels, 'preds': preds, 'probs': probs,
        'accuracy': acc, 'f1_macro': f1,
    }

    # ── Summary table ──
    print(f"\n{'='*75}")
    print(f"{'Model':<25} {'Accuracy':<12} {'F1-Macro':<12} {'Params (M)':<12}")
    print(f"{'─'*75}")
    for name in list(cfg.baseline_models.keys()) + ['NeuroFusionNet']:
        r = test_results[name]
        n_params = sum(p.numel() for p in all_models[name].parameters()) / 1e6
        print(f"{name:<25} {r['accuracy']:<12.4f} {r['f1_macro']:<12.4f} {n_params:<12.1f}")
    print(f"{'='*75}")
