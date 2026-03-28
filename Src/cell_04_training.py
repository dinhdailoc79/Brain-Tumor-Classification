import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from cell_03_models import *

# %% Cell 4: Training Engine — Baselines + NeuroFusionNet
# ════════════════════════════════════════════════════════════════════════
# Two-phase training with EMA, SWA, Mixup/CutMix, and multi-task loss
# ════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
# 1. BASELINE TRAINER
# ════════════════════════════════════════════════════════════════════════

class BaselineTrainer:
    """Two-phase training for baseline models (EfficientNet, ConvNeXt, ViT, Swin)."""

    def __init__(self, model, cfg, device, class_counts=None):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device

        # Loss
        alpha = get_focal_loss_alpha(train_df_final) if class_counts is None else None
        if alpha is None and class_counts is not None:
            total = sum(class_counts.values())
            alpha = [total / (cfg.num_classes * class_counts.get(i, 1))
                     for i in range(cfg.num_classes)]
            max_a = max(alpha)
            alpha = [a / max_a for a in alpha]
        self.criterion = FocalLoss(alpha=alpha, gamma=2.0,
                                    label_smoothing=cfg.label_smoothing)

        # AMP
        self.amp_dtype = torch.bfloat16 if cfg.amp_dtype == 'bfloat16' else torch.float16
        self.use_scaler = (cfg.amp_dtype == 'float16')
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_scaler)

        # EMA (faster tracking for small dataset)
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.eval()
        self.ema_decay = 0.995  # Faster than 0.999 for ~7k dataset

        # History
        self.history = {'train_loss': [], 'val_loss': [],
                        'train_acc': [], 'val_acc': [],
                        'train_f1': [], 'val_f1': [], 'lr': []}
        self.best_val_acc = 0.0
        self.best_model_state = None
        self.no_improve = 0

    def _update_ema(self):
        with torch.no_grad():
            for ep, mp in zip(self.ema_model.parameters(), self.model.parameters()):
                ep.data.mul_(self.ema_decay).add_(mp.data, alpha=1 - self.ema_decay)

    def _train_epoch(self, loader, optimizer, epoch):
        self.model.train()
        total_loss, correct, total_n = 0.0, 0, 0
        all_preds, all_labels = [], []
        pbar = tqdm(loader, desc=f"  Train Ep{epoch+1:02d}", leave=False)

        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            # Mixup/CutMix
            apply_mix = np.random.random() < self.cfg.mix_prob
            if apply_mix and epoch >= self.cfg.phase1_epochs:
                if np.random.random() < 0.5:
                    images, ya, yb, lam = mixup_data(images, labels, self.cfg.mixup_alpha)
                else:
                    images, ya, yb, lam = cutmix_data(images, labels, self.cfg.cutmix_alpha)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                logits = self.model(images)
                if apply_mix and epoch >= self.cfg.phase1_epochs:
                    loss = lam * self.criterion(logits, ya) + (1 - lam) * self.criterion(logits, yb)
                else:
                    loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clip)
            self.scaler.step(optimizer)
            self.scaler.update()
            self._update_ema()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total_n += images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})

        return (total_loss / total_n, correct / total_n,
                f1_score(all_labels, all_preds, average='macro', zero_division=0))

    @torch.no_grad()
    def _validate(self, loader):
        self.ema_model.eval()
        total_loss, correct, total_n = 0.0, 0, 0
        all_preds, all_labels = [], []

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                logits = self.ema_model(images)
                loss = F.cross_entropy(logits, labels)
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total_n += images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        return (total_loss / total_n, correct / total_n,
                f1_score(all_labels, all_preds, average='macro'))

    def fit(self, train_loader, val_loader):
        t0 = time.time()

        # ═══ Phase 1: Frozen backbone ═══
        print("🔒 Phase 1: Training head (backbone frozen)")
        freeze_backbone(self.model)
        opt = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                          lr=self.cfg.phase1_lr, weight_decay=self.cfg.weight_decay)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.phase1_epochs)

        for ep in range(self.cfg.phase1_epochs):
            tl, ta, tf = self._train_epoch(train_loader, opt, ep)
            vl, va, vf = self._validate(val_loader)
            sched.step()
            self._log(ep, tl, vl, ta, va, tf, vf, opt.param_groups[0]['lr'])
            self._check(va)

        # ═══ Phase 2: Full fine-tune ═══
        print("\n🔓 Phase 2: Full fine-tuning")
        unfreeze_all(self.model)

        backbone_params, head_params = [], []
        for name, p in self.model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(p)
            else:
                head_params.append(p)

        opt = optim.AdamW([
            {'params': backbone_params, 'lr': self.cfg.phase2_backbone_lr, 'weight_decay': 0.05},
            {'params': head_params, 'lr': self.cfg.phase2_head_lr, 'weight_decay': 0.01},
        ])
        warmup = optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=3)
        cosine = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.phase2_epochs - 3)
        sched = optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[3])

        swa_model = AveragedModel(self.model)
        swa_start = 15
        self.no_improve = 0

        for ep in range(self.cfg.phase2_epochs):
            offset = self.cfg.phase1_epochs
            tl, ta, tf = self._train_epoch(train_loader, opt, ep + offset)
            vl, va, vf = self._validate(val_loader)
            sched.step()
            self._log(ep + offset, tl, vl, ta, va, tf, vf, opt.param_groups[0]['lr'])
            self._check(va)

            if ep >= swa_start:
                swa_model.update_parameters(self.model)

            if self.no_improve >= self.cfg.patience:
                print(f"   ⏹ Early stopping at epoch {ep + offset + 1}")
                break

        print(f"\n⏱ Training: {(time.time()-t0)/60:.1f} min | Best val acc: {self.best_val_acc:.4f}")
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        return self.history

    def _log(self, ep, tl, vl, ta, va, tf, vf, lr):
        self.history['train_loss'].append(tl)
        self.history['val_loss'].append(vl)
        self.history['train_acc'].append(ta)
        self.history['val_acc'].append(va)
        self.history['train_f1'].append(tf)
        self.history['val_f1'].append(vf)
        self.history['lr'].append(lr)
        print(f"   Ep {ep+1:02d} │ TrA:{ta:.4f} TrF1:{tf:.4f} │ "
              f"VaA:{va:.4f} VaF1:{vf:.4f} │ LR:{lr:.1e}")

    def _check(self, va):
        if va > self.best_val_acc:
            self.best_val_acc = va
            self.best_model_state = copy.deepcopy(self.model.state_dict())
            self.no_improve = 0
        else:
            self.no_improve += 1


# ════════════════════════════════════════════════════════════════════════
# 2. NEUROFUSIONNET TRAINER (Multi-Task Loss)
# ════════════════════════════════════════════════════════════════════════

class NeuroFusionTrainer:
    """Trainer for NeuroFusionNet with multi-task loss:
      L_total = L_main + α·L_tumor + β·L_fine + γ·L_contrast
    """

    def __init__(self, model, cfg, device, class_counts=None):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device

        # Focal Loss weights
        if class_counts:
            total = sum(class_counts.values())
            alpha = [total / (cfg.num_classes * class_counts.get(i, 1))
                     for i in range(cfg.num_classes)]
            max_a = max(alpha)
            alpha = [a / max_a for a in alpha]
        else:
            alpha = get_focal_loss_alpha(train_df_final)

        self.loss_main = LargeMarginFocalLoss(
            alpha=alpha, gamma=2.0, margin_scale=0.3,
            label_smoothing=0.1, num_classes=cfg.num_classes)
        self.loss_tumor = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.loss_fine = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.loss_con = SupConLoss(temperature=0.07)

        # AMP
        self.amp_dtype = torch.bfloat16 if cfg.amp_dtype == 'bfloat16' else torch.float16
        self.use_scaler = (cfg.amp_dtype == 'float16')
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_scaler)

        # EMA
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.eval()

        # History
        self.history = {'train_loss': [], 'val_loss': [],
                        'train_acc': [], 'val_acc': [],
                        'train_f1': [], 'val_f1': [], 'lr': []}
        self.best_val_acc = 0.0
        self.best_model_state = None
        self.no_improve = 0

    def _update_ema(self):
        with torch.no_grad():
            for ep, mp in zip(self.ema_model.parameters(), self.model.parameters()):
                ep.data.mul_(cfg.ema_decay).add_(mp.data, alpha=1 - cfg.ema_decay)

    def _multi_task_loss(self, outputs, labels):
        logits = outputs['logits']
        loss_main = self.loss_main(logits, labels)

        tumor_labels = (labels != 2).long()  # notumor=0, tumor=1
        loss_tumor = self.loss_tumor(outputs['tumor_logits'], tumor_labels)

        gm_mask = (labels == 0) | (labels == 1)
        if gm_mask.sum() > 1:
            loss_fine = self.loss_fine(outputs['fine_logits'][gm_mask], labels[gm_mask])
        else:
            loss_fine = torch.tensor(0.0, device=labels.device)

        loss_con = self.loss_con(outputs['embeddings'], labels)

        total = (cfg.w_main * loss_main + cfg.w_tumor * loss_tumor +
                 cfg.w_fine * loss_fine + cfg.w_contrastive * loss_con)
        return total

    def _train_epoch(self, loader, optimizer, epoch):
        self.model.train()
        total_loss, correct, total_n = 0.0, 0, 0
        all_preds, all_labels = [], []
        pbar = tqdm(loader, desc=f"  Train Ep{epoch+1:02d}", leave=False)

        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                outputs = self.model(images, return_all=True)
                loss = self._multi_task_loss(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip)
            self.scaler.step(optimizer)
            self.scaler.update()
            self._update_ema()

            total_loss += loss.item() * images.size(0)
            preds = outputs['logits'].argmax(1)
            correct += (preds == labels).sum().item()
            total_n += images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})

        return (total_loss / total_n, correct / total_n,
                f1_score(all_labels, all_preds, average='macro', zero_division=0))

    @torch.no_grad()
    def _validate(self, loader):
        self.ema_model.eval()
        total_loss, correct, total_n = 0.0, 0, 0
        all_preds, all_labels = [], []

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                logits = self.ema_model(images, return_all=False)
                loss = F.cross_entropy(logits, labels)
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total_n += images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        return (total_loss / total_n, correct / total_n,
                f1_score(all_labels, all_preds, average='macro'))

    def fit(self, train_loader, val_loader):
        t0 = time.time()

        # ═══ Phase 1: Freeze backbones ═══
        print("🔒 Phase 1: Training fusion + expert heads (backbones frozen)")
        freeze_backbone(self.model)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   Trainable: {trainable/1e6:.2f}M params")

        opt = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-3, weight_decay=0.01)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5, eta_min=1e-5)

        for ep in range(5):
            tl, ta, tf = self._train_epoch(train_loader, opt, ep)
            vl, va, vf = self._validate(val_loader)
            sched.step()
            self._log(ep, tl, vl, ta, va, tf, vf, opt.param_groups[0]['lr'])
            self._check(va)

        # ═══ Phase 2: Full fine-tune ═══
        print("\n🔓 Phase 2: Full fine-tuning (Multi-Task Loss)")
        unfreeze_all(self.model)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   Trainable: {trainable/1e6:.2f}M params")

        backbone_params, head_params = [], []
        for name, p in self.model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(p)
            else:
                head_params.append(p)

        opt = optim.AdamW([
            {'params': backbone_params, 'lr': 2e-5, 'weight_decay': 0.05},
            {'params': head_params, 'lr': 5e-4, 'weight_decay': 0.01},
        ])
        warmup = optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=3)
        cosine = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=22, eta_min=1e-7)
        sched = optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[3])

        swa_model = AveragedModel(self.model)
        swa_start = 15
        self.no_improve = 0

        for ep in range(25):
            tl, ta, tf = self._train_epoch(train_loader, opt, ep + 5)
            vl, va, vf = self._validate(val_loader)
            sched.step()
            self._log(ep + 5, tl, vl, ta, va, tf, vf, opt.param_groups[0]['lr'])
            self._check(va)

            if ep >= swa_start:
                swa_model.update_parameters(self.model)

            if self.no_improve >= 8:
                print(f"   ⏹ Early stopping at epoch {ep + 6}")
                break

        print(f"\n⏱ Training: {(time.time()-t0)/60:.1f} min | Best val acc: {self.best_val_acc:.4f}")
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        return self.history

    def _log(self, ep, tl, vl, ta, va, tf, vf, lr):
        self.history['train_loss'].append(tl)
        self.history['val_loss'].append(vl)
        self.history['train_acc'].append(ta)
        self.history['val_acc'].append(va)
        self.history['train_f1'].append(tf)
        self.history['val_f1'].append(vf)
        self.history['lr'].append(lr)
        print(f"   Ep {ep+1:02d} │ TrA:{ta:.4f} TrF1:{tf:.4f} │ "
              f"VaA:{va:.4f} VaF1:{vf:.4f} │ LR:{lr:.1e}")

    def _check(self, va):
        if va > self.best_val_acc:
            self.best_val_acc = va
            self.best_model_state = copy.deepcopy(self.model.state_dict())
            self.no_improve = 0
        else:
            self.no_improve += 1


# ════════════════════════════════════════════════════════════════════════
# 3. EVALUATION FUNCTION
# ════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_model(model, loader, device):
    """Full evaluation returning labels, predictions, and probabilities."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device)
        with torch.amp.autocast('cuda'):
            output = model(images)
            logits = output['logits'] if isinstance(output, dict) else output
            probs = F.softmax(logits, dim=1)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    setup_data()
    print("✅ Training engine ready!")
