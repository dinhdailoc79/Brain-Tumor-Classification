import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from cell_02_data import *

# %% Cell 3: Baseline Models + NeuroFusionNet Architecture
# ════════════════════════════════════════════════════════════════════════
# DESIGN (First Principles): Architecture mirrors clinical workflow:
#   Expert 1: "Is there a tumor?" (binary) — like radiologist's first scan
#   Expert 2: "What type?" (4-class)  — detailed classification
#   Expert 3: "Glioma or Meningioma?" (fine-grained) — hardest distinction
#   Uncertainty: "How sure am I?" — when to call for consultation
#
# TERMINOLOGY NOTE (for paper):
#   - The 3 Expert Heads are MULTI-TASK HEADS sharing the same fused features,
#     NOT independent Mixture-of-Experts (MoE). They provide complementary
#     classification signals at different granularities.
#   - Cross-Attention Fusion operates on CLS-level embeddings (1 token each),
#     not spatial feature maps. It is "CLS-level attention-gated fusion".
#   - Temperature Scaling is applied POST-HOC on the validation set
#     (Guo et al., ICML 2017), NOT as a learnable parameter during training.
# ════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════
# 1. BASELINE MODEL FACTORY
# ════════════════════════════════════════════════════════════════════════

class BaselineModel(nn.Module):
    """Wraps any timm model with a custom classification head."""

    def __init__(self, model_key: str, num_classes: int = 4,
                 pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(model_key, pretrained=pretrained, num_classes=0)
        self.num_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(self.num_features),
            nn.Dropout(dropout),
            nn.Linear(self.num_features, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def get_embeddings(self, x):
        return self.backbone(x)


def create_baseline(model_key: str, num_classes: int = 4) -> BaselineModel:
    model = BaselineModel(model_key, num_classes)
    print(f"   Created {model_key}: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    return model


def freeze_backbone(model):
    """Freeze all backbone parameters."""
    if hasattr(model, 'backbone'):
        for p in model.backbone.parameters():
            p.requires_grad = False
    elif hasattr(model, 'backbone_global'):
        for p in model.backbone_global.parameters():
            p.requires_grad = False
        for p in model.backbone_local.parameters():
            p.requires_grad = False


def unfreeze_all(model):
    """Unfreeze all parameters."""
    for p in model.parameters():
        p.requires_grad = True


# ════════════════════════════════════════════════════════════════════════
# 2. LOSS FUNCTIONS
# ════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """Standard Focal Loss with label smoothing."""

    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none',
                             label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce

        if self.alpha is not None:
            at = self.alpha.to(logits.device)[targets]
            focal = at * focal

        return focal.mean()


class LargeMarginFocalLoss(nn.Module):
    """Focal Loss + class-dependent large margins.
    Minority classes get wider margins → model pays extra attention.
    """

    def __init__(self, alpha=None, gamma=2.0, margin_scale=0.3,
                 label_smoothing=0.1, num_classes=4):
        super().__init__()
        self.gamma = gamma
        self.margin_scale = margin_scale
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits, targets):
        margins = torch.zeros_like(logits)
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            for i in range(self.num_classes):
                mask = (targets == i)
                margins[mask, i] = -self.margin_scale * alpha[i]

        ce = F.cross_entropy(logits + margins, targets, reduction='none',
                             label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce

        if self.alpha is not None:
            at = self.alpha.to(logits.device)[targets]
            focal = at * focal

        return focal.mean()


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).
    Pulls same-class embeddings together in feature space.
    Critical for separating glioma ↔ meningioma clusters.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        B = features.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=features.device)

        sim = torch.matmul(features, features.T) / self.temperature
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float()
        eye = torch.eye(B, device=features.device)
        mask = mask - eye

        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        exp_sim = torch.exp(sim) * (1 - eye)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        pos_count = torch.clamp(mask.sum(dim=1), min=1)
        mean_log_prob = (mask * log_prob).sum(dim=1) / pos_count

        return -mean_log_prob.mean()


# ════════════════════════════════════════════════════════════════════════
# 3. CLS-LEVEL ATTENTION-GATED FUSION MODULE
# ════════════════════════════════════════════════════════════════════════
# NOTE: This is NOT spatial cross-attention (ViLT/BEiT-3 style).
# Both backbones output a single CLS token (1D vector), so attention
# operates on 2 tokens, not spatial feature maps. The mechanism allows
# the global backbone (Swin) to "query" the local backbone (ConvNeXt)
# for complementary information before fusion.
# ════════════════════════════════════════════════════════════════════════

class CrossAttentionFusion(nn.Module):
    """Fuses features from dual backbones via cross-attention.

    Analogy (Aviation CRM):
      - Pilot (Swin-V2) sees the big picture (global context)
      - Co-pilot (ConvNeXt) checks details (local texture)
      - Cross-attention = they discuss before making a decision
    """

    def __init__(self, dim_a, dim_b, hidden_dim=512, num_heads=8):
        super().__init__()
        self.proj_a = nn.Linear(dim_a, hidden_dim)
        self.proj_b = nn.Linear(dim_b, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),  # Increased for MC Dropout variance
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, feat_a, feat_b):
        a = self.proj_a(feat_a).unsqueeze(1)  # (B, 1, H)
        b = self.proj_b(feat_b).unsqueeze(1)  # (B, 1, H)

        # Cross-attention: Global queries Local's information
        attn_out, _ = self.cross_attn(a, b, b)
        a = self.norm(a + attn_out)  # residual + norm

        fused = torch.cat([a.squeeze(1), b.squeeze(1)], dim=-1)
        fused = self.mlp(fused)
        fused = self.out_norm(fused)
        return fused


# ════════════════════════════════════════════════════════════════════════
# 4. MULTI-TASK EXPERT HEADS + GATING
# ════════════════════════════════════════════════════════════════════════
# IMPORTANT: These heads share the SAME fused feature vector (512-dim).
# They are NOT independent experts with different views of the data.
# Think of them as multi-task heads providing different granularity
# classifications from the same representation, combined via gating.
# ════════════════════════════════════════════════════════════════════════

class ExpertHead(nn.Module):
    """Lightweight expert classifier head with MC Dropout support.
    Dropout rate is intentionally high (0.4) for meaningful MC variance.
    """

    def __init__(self, in_dim, num_classes, dropout=0.4):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),         # Primary MC Dropout point
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout * 0.6),   # Secondary MC Dropout point
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(x)


# ════════════════════════════════════════════════════════════════════════
# 5. NEUROFUSIONNET — MAIN ARCHITECTURE
# ════════════════════════════════════════════════════════════════════════

class NeuroFusionNet(nn.Module):
    """
    NeuroFusionNet: Hierarchical Multi-Task Network with
    Dual-Backbone CLS-Level Attention Fusion for Brain Tumor Classification.

    Architecture (mirrors clinical workflow):
        Input → [Swin-V2-S (global)] + [ConvNeXt-Base (local)]
              → CLS-Level Attention Fusion (512-dim)
              → Expert Head 1: Tumor Detector (binary)
              → Expert Head 2: 4-class Classifier
              → Expert Head 3: Fine-Grained Glioma vs Meningioma
              → Gated Multi-Task Combination
              → Output (temperature scaling applied post-hoc)

    KEY DESIGN NOTES:
        - Expert heads SHARE the same fused features (multi-task, not MoE)
        - Temperature scaling is NOT a learnable param — it is calibrated
          post-hoc on the validation set (Guo et al., ICML 2017)
        - Gating constants were derived from clinical risk analysis:
          notumor boost=3.0 (high penalty for false negatives),
          fine-grained scale=2.0 (moderate redistribution)

    Uncertainty support:
        - MC Dropout: enable_mc_dropout() keeps dropout ON during inference
        - Forward T times → get prediction distribution → uncertainty
    """

    # Class ordering for gating logic (must match cfg.class_names)
    CLASS_ORDER = ['glioma', 'meningioma', 'notumor', 'pituitary']

    # Gating constants (derived from clinical risk analysis):
    #   NOTUMOR_BOOST: When tumor detector says "no tumor", amplify notumor
    #     logit by this factor. Set HIGH (3.0) because missing a tumor
    #     (false negative) is catastrophic — patient goes home untreated.
    #   FINE_SCALE: How much to redistribute glioma/meningioma logits
    #     based on the fine-grained expert. Moderate (2.0) because the
    #     fine-grained expert shares the same features.
    NOTUMOR_BOOST_FACTOR = 3.0
    FINE_SCALE_FACTOR = 2.0

    def __init__(self, num_classes=4, pretrained=True,
                 hidden_dim=512, dropout=0.3,
                 class_names=None):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self._mc_dropout = False  # MC Dropout flag

        # Class name mapping for gating (default matches Config)
        self._class_names = class_names or self.CLASS_ORDER
        self._notumor_idx = self._class_names.index('notumor')
        self._glioma_idx = self._class_names.index('glioma')
        self._menin_idx = self._class_names.index('meningioma')

        # ── Dual Backbones ──
        self.backbone_global = timm.create_model(
            'swinv2_small_window8_256', pretrained=pretrained, num_classes=0
        )
        self.backbone_local = timm.create_model(
            'convnext_base', pretrained=pretrained, num_classes=0
        )
        dim_global = self.backbone_global.num_features  # 768
        dim_local = self.backbone_local.num_features     # 1024

        # ── CLS-Level Attention Fusion ──
        self.fusion = CrossAttentionFusion(dim_global, dim_local, hidden_dim)

        # ── Contrastive Projector ──
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 128),
        )

        # ── Multi-Task Expert Heads (shared fused features) ──
        self.expert_tumor = ExpertHead(hidden_dim, 2, dropout)    # binary
        self.expert_type = ExpertHead(hidden_dim, num_classes, dropout)  # 4-class
        self.expert_fine = ExpertHead(hidden_dim, 2, dropout)     # glioma vs menin

        # ── Gating Network ──
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

        # ── Temperature ──
        # Fixed at 1.0 during training. Proper post-hoc calibration
        # (Platt Scaling / Temperature Scaling) is performed AFTER training
        # on the validation set in cell_06. See: Guo et al., ICML 2017.
        self.register_buffer('temperature', torch.ones(1))

        # Compatibility with pipeline
        self.num_features = hidden_dim

    def enable_mc_dropout(self):
        """Enable MC Dropout for uncertainty estimation.
        IMMEDIATELY forces all Dropout layers into train mode.
        """
        self._mc_dropout = True
        # Force dropout active RIGHT NOW (not deferred to train())
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def disable_mc_dropout(self):
        """Disable MC Dropout, return to normal eval behavior."""
        self._mc_dropout = False
        self.eval()  # Re-apply eval to all modules including dropout

    def train(self, mode=True):
        """Override train() to support MC Dropout."""
        super().train(mode)
        if not mode and self._mc_dropout:
            # In eval mode but MC Dropout enabled → keep dropout active
            for m in self.modules():
                if isinstance(m, nn.Dropout):
                    m.train()
        return self

    def forward(self, x, return_all=False):
        """
        Args:
            x: input (B, 3, H, W)
            return_all: return dict for multi-task loss

        Returns:
            if return_all: dict with logits, tumor_logits, fine_logits,
                           embeddings, gate_weights
            else: final 4-class logits (B, 4)
        """
        # ── Dual backbone features ──
        feat_global = self.backbone_global(x)  # (B, 768)
        feat_local = self.backbone_local(x)    # (B, 1024)

        # ── Cross-Attention Fusion ──
        fused = self.fusion(feat_global, feat_local)  # (B, 512)

        # ── Expert Predictions ──
        tumor_logits = self.expert_tumor(fused)   # (B, 2)
        type_logits = self.expert_type(fused)      # (B, 4)
        fine_logits = self.expert_fine(fused)       # (B, 2)

        # ── Gated Multi-Task Combination ──
        gate_weights = torch.softmax(self.gate(fused), dim=-1)  # (B, 3)

        tumor_prob = torch.softmax(tumor_logits, dim=-1)
        notumor_boost = tumor_prob[:, 0:1]  # P(no tumor)

        fine_prob = torch.softmax(fine_logits, dim=-1)

        final_logits = type_logits.clone()

        # Expert Head 1 → boost notumor logit when detector says "no tumor"
        # Uses class-name-based index (not hardcoded) for robustness
        final_logits[:, self._notumor_idx] += (
            gate_weights[:, 0] * self.NOTUMOR_BOOST_FACTOR * notumor_boost.squeeze(-1)
        )

        # Expert Head 3 → redistribute glioma/meningioma logits
        gm_weight = gate_weights[:, 2]
        final_logits[:, self._glioma_idx] += (
            gm_weight * (fine_prob[:, 0] - 0.5) * self.FINE_SCALE_FACTOR
        )
        final_logits[:, self._menin_idx] += (
            gm_weight * (fine_prob[:, 1] - 0.5) * self.FINE_SCALE_FACTOR
        )

        # Temperature scaling (post-hoc calibrated, default=1.0 during training)
        final_logits = final_logits / self.temperature

        if return_all:
            embeddings = self.projector(fused)
            return {
                'logits': final_logits,
                'tumor_logits': tumor_logits,
                'fine_logits': fine_logits,
                'embeddings': embeddings,
                'gate_weights': gate_weights,
                'fused_features': fused,
            }
        return final_logits

    def get_embeddings(self, x):
        """Extract fused embeddings for t-SNE visualization."""
        feat_global = self.backbone_global(x)
        feat_local = self.backbone_local(x)
        return self.fusion(feat_global, feat_local)


# ════════════════════════════════════════════════════════════════════════
# 6. HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════

def get_focal_loss_alpha(df, num_classes=4):
    """Compute inverse-frequency class weights for Focal Loss."""
    counts = Counter(df['label_idx'].tolist())
    total = sum(counts.values())
    alpha = [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
    max_a = max(alpha)
    alpha = [a / max_a for a in alpha]
    return alpha


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation on batch."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix augmentation on batch."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    index = torch.randperm(x.size(0), device=x.device)

    _, _, H, W = x.shape
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    return mixed_x, y, y[index], lam


# ════════════════════════════════════════════════════════════════════════
# INSTANTIATION CHECK
# ════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    setup_data()

    print("🧠 Testing NeuroFusionNet instantiation...")
    _test_model = NeuroFusionNet(num_classes=4, pretrained=False, hidden_dim=512)
    _total = sum(p.numel() for p in _test_model.parameters())
    _global = sum(p.numel() for p in _test_model.backbone_global.parameters())
    _local = sum(p.numel() for p in _test_model.backbone_local.parameters())
    _heads = _total - _global - _local
    print(f"   Total: {_total/1e6:.1f}M | Swin-V2-S: {_global/1e6:.1f}M | "
          f"ConvNeXt-B: {_local/1e6:.1f}M | Fusion+Experts: {_heads/1e6:.1f}M")
    # Test forward pass
    _dummy = torch.randn(2, 3, 256, 256)
    _out = _test_model(_dummy, return_all=True)
    print(f"   Forward pass OK: logits={_out['logits'].shape}, "
          f"gate_weights={_out['gate_weights'].shape}")
    # Test MC Dropout
    _test_model.eval()
    _test_model.enable_mc_dropout()
    _out1 = _test_model(_dummy)
    _out2 = _test_model(_dummy)
    print(f"   MC Dropout OK: outputs differ = {not torch.allclose(_out1, _out2)}")
    _test_model.disable_mc_dropout()
    del _test_model, _dummy, _out, _out1, _out2
    gc.collect()
    print("✅ All model components verified!\n")
