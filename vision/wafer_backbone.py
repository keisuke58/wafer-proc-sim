"""
Shared wafer-image CNN backbone — pretrain once, transfer everywhere.

The convolutional stack is identical to the one inlined in
``vision.kerf_quality_classifier.train_cnn`` (8→16→32 channels, ending in
``AdaptiveAvgPool2d(1)``).  Because the backbone is terminated by global
average pooling it is **input-size agnostic**: a backbone pretrained on
64×64 WM-811K wafer maps drops straight into the 128×128 kerf classifier
or the 52×52 MixedWM38 benchmark with no shape surgery.

Transfer story used across this repo:
    WM-811K (811k real fab wafer maps, 9 classes)
        └─ pretrain_wm811k.py  ──►  results/wm811k_backbone.pt
                                        │
              ┌─────────────────────────┼────────────────────────────┐
              ▼                          ▼                            ▼
    kerf_quality_classifier      mixedwm38_benchmark         (future dicing
      (3-class chipping,           (8-way multilabel,           defect maps)
       --pretrained)                --init-backbone)

Usage
-----
    from vision.wafer_backbone import WaferBackbone, WaferClassifier
    net = WaferClassifier(n_classes=9)          # backbone + linear head
    ...                                          # train
    net.backbone.save("results/wm811k_backbone.pt")
    # later, in another task:
    clf = WaferClassifier(n_classes=3)
    clf.backbone.load("results/wm811k_backbone.pt")   # warm start
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

# torch is a hard dependency of the DL modules in this repo (see
# ml/disco_blade_wear_dl.py); import eagerly so failures are loud.
import torch
import torch.nn as nn
import torch.nn.functional as F

BACKBONE_FEATURES = 32  # output dim of the global-average-pooled backbone
DEFAULT_BACKBONE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "wm811k_backbone.pt"
)


class WaferBackbone(nn.Module):
    """3-layer conv feature extractor → (N, 32) embedding. Size-agnostic."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N,1,H,W) -> (N,32)
        return self.features(x)

    def save(self, path: str = DEFAULT_BACKBONE_PATH) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        return path

    def load(self, path: str = DEFAULT_BACKBONE_PATH,
             map_location: str = "cpu") -> "WaferBackbone":
        self.load_state_dict(torch.load(path, map_location=map_location))
        return self


class WaferClassifier(nn.Module):
    """Backbone + linear head. ``multilabel`` switches loss/activation."""

    def __init__(self, n_classes: int, multilabel: bool = False):
        super().__init__()
        self.backbone = WaferBackbone()
        self.head = nn.Linear(BACKBONE_FEATURES, n_classes)
        self.multilabel = multilabel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ── image helpers ────────────────────────────────────────────────────────────

def to_grayscale_tensor(X: np.ndarray, size: Optional[int] = None,
                        vmax: Optional[float] = None) -> torch.Tensor:
    """
    (N, H, W) uint/int/float array → (N, 1, size, size) float tensor in [0, 1].

    ``vmax`` sets the normalisation denominator (defaults to the array max, or
    255 for pixel images / 2 for {0,1,2} wafer maps — pass explicitly to be
    safe).  ``size`` resizes via nearest-neighbour interpolation; ``None``
    keeps the native resolution (only valid when every image is same-sized).
    """
    t = torch.from_numpy(np.ascontiguousarray(X)).float()[:, None, :, :]
    if vmax is None:
        vmax = float(t.max()) or 1.0
    t = t / vmax
    if size is not None:
        t = F.interpolate(t, size=(size, size), mode="nearest")
    return t


def train_classifier(
    model: WaferClassifier,
    X_train: torch.Tensor, y_train: torch.Tensor,
    X_val: Optional[torch.Tensor] = None, y_val: Optional[torch.Tensor] = None,
    *, epochs: int = 10, batch_size: int = 128, lr: float = 2e-3,
    class_weight: Optional[torch.Tensor] = None, device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """
    Generic mini-batch trainer. Multiclass uses CrossEntropy (int labels);
    multilabel uses BCEWithLogits (float multi-hot labels).
    Returns a history dict.
    """
    model = model.to(device)
    if model.multilabel:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weight)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"epoch": [], "train_loss": [], "val_metric": []}

    Xtr, ytr = X_train.to(device), y_train.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(Xtr))
        losses = []
        for i in range(0, len(Xtr), batch_size):
            idx = order[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        val_metric = float("nan")
        if X_val is not None and len(X_val):
            model.eval()
            with torch.no_grad():
                logits = model(X_val.to(device))
                if model.multilabel:
                    pred = (torch.sigmoid(logits) > 0.5).float()
                    val_metric = float((pred == y_val.to(device)).float().mean())
                else:
                    val_metric = float(
                        (logits.argmax(1) == y_val.to(device)).float().mean())
        history["epoch"].append(epoch)
        history["train_loss"].append(float(np.mean(losses)))
        history["val_metric"].append(val_metric)
        if verbose:
            print(f"    epoch {epoch:2d}: train_loss={np.mean(losses):.4f} "
                  f"val={val_metric:.3f}")
    return history


@torch.no_grad()
def predict(model: WaferClassifier, X: torch.Tensor,
            device: str = "cpu", batch_size: int = 256) -> np.ndarray:
    """Batched inference. Returns int class ids (multiclass) or multi-hot 0/1."""
    model = model.to(device).eval()
    outs = []
    for i in range(0, len(X), batch_size):
        logits = model(X[i:i + batch_size].to(device))
        if model.multilabel:
            outs.append((torch.sigmoid(logits) > 0.5).int().cpu().numpy())
        else:
            outs.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(outs)
