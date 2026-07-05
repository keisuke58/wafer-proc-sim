#!/usr/bin/env python3
"""adc_active_learning.py — label-efficient deep ADC: which images to label?

The bottleneck of deploying an AI defect classifier in a fab is not the model
— it is LABELS. Every labeled kerf image costs a process engineer's minutes,
and defect classes are rare. `ml/active_learning.py` does Bayesian
experimental design on tabular process parameters; this module does the deep
image version: a CNN kerf-grade classifier (A/B/C from
`vision/kerf_quality_classifier.make_grade_dataset`) trained under a LABEL
BUDGET, comparing three labeling strategies over identical seeds:

  RANDOM      label whatever comes off the line (the default everywhere)
  ENTROPY     label the images the current model is least sure about
  ENT+DIV     entropy shortlist, then k-center diversity on the CNN's own
              embedding — avoids buying 30 near-duplicates of one confusion

Deliverable is the curve a manager actually asks for: accuracy vs number of
labels, and the label count each strategy needs to reach the 90 % line.

torch (CPU) + numpy + matplotlib.  Run:  python3 vision/adc_active_learning.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vision.kerf_quality_classifier import make_grade_dataset  # noqa: E402

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "results")
ASSETS = os.path.join(os.path.dirname(__file__), "..", "vision", "cpp",
                      "docs", "assets")
RESULTS_JSON = os.path.join(os.path.dirname(__file__), "..", "ml",
                            "adc_active_learning_results.json")

IMG = 64            # working resolution (downsampled from 128)
INIT_LABELS = 24
BATCH = 24
ROUNDS = 7          # final budget = 24 + 7*24 = 192 labels
TARGET_ACC = 0.90


def _build(seed):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.feat = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 48, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(4),
            )
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(48 * 16, 64),
                                      nn.ReLU(), nn.Linear(64, 3))

        def forward(self, x, embed=False):
            z = self.feat(x)
            if embed:
                return z.flatten(1)
            return self.head(z)

    return Net()


def _train(model, X, y, epochs=30, seed=0):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed + 1)
    opt = torch.optim.Adam(model.parameters(), lr=1.5e-3)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(X)
    yt = torch.tensor(y)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 32):
            idx = perm[i:i + 32]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    return model


def _probs(model, X, bs=256):
    import torch
    with torch.no_grad():
        out = []
        for i in range(0, len(X), bs):
            out.append(torch.softmax(model(torch.tensor(X[i:i + bs])), 1))
        return torch.cat(out).numpy()


def _embed(model, X, bs=256):
    import torch
    with torch.no_grad():
        out = []
        for i in range(0, len(X), bs):
            out.append(model(torch.tensor(X[i:i + bs]), embed=True))
        return torch.cat(out).numpy()


def _select(strategy, model, X_pool, pool_idx, k, rng):
    if strategy == "random":
        return rng.choice(pool_idx, size=k, replace=False)
    p = _probs(model, X_pool[pool_idx])
    ent = -(p * np.log(p + 1e-12)).sum(axis=1)
    if strategy == "entropy":
        return pool_idx[np.argsort(ent)[-k:]]
    # entropy shortlist (4k) then k-center greedy on embeddings
    short = pool_idx[np.argsort(ent)[-4 * k:]]
    emb = _embed(model, X_pool[short])
    chosen = [int(np.argmax(ent[np.argsort(ent)[-4 * k:]]))]
    d = np.linalg.norm(emb - emb[chosen[0]], axis=1)
    while len(chosen) < k:
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(emb - emb[nxt], axis=1))
    return short[np.asarray(chosen)]


def prepare_data(n_per_class=260, seed=0):
    X, y = make_grade_dataset(n_per_class=n_per_class, H=128, W=128,
                              seed=seed)
    X = X[:, ::2, ::2]                                   # 64x64
    X = (X / 255.0).astype(np.float32)[:, None, :, :]    # NCHW
    mu, sd = X.mean(), X.std()
    X = (X - mu) / sd
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    X, y = X[order], y[order]
    n_test = max(30, int(0.27 * len(X)))
    return (X[n_test:], y[n_test:]), (X[:n_test], y[:n_test])


def run_strategy(strategy, X, y, X_te, y_te, seed):
    rng = np.random.default_rng(seed)
    labeled = list(rng.choice(len(X), size=INIT_LABELS, replace=False))
    accs, budgets = [], []
    model = None
    for rnd in range(ROUNDS + 1):
        model = _build(seed + rnd)
        model = _train(model, X[labeled], y[labeled], seed=seed + rnd)
        acc = float((np.argmax(_probs(model, X_te), 1) == y_te).mean())
        accs.append(acc)
        budgets.append(len(labeled))
        if rnd == ROUNDS:
            break
        pool = np.setdiff1d(np.arange(len(X)), labeled)
        picked = _select(strategy, model, X, pool, BATCH, rng)
        labeled.extend(int(i) for i in picked)
    return np.asarray(budgets), np.asarray(accs)


def labels_to_target(budgets, accs, target=TARGET_ACC):
    for b, a in zip(budgets, accs):
        if a >= target:
            return int(b)
    return None


def run(seeds=(0, 1, 2)):
    (X, y), (X_te, y_te) = prepare_data()
    out = {}
    for strat in ("random", "entropy", "entdiv"):
        curves = []
        for s in seeds:
            b, a = run_strategy(strat, X, y, X_te, y_te, seed=100 + s)
            curves.append(a)
        curves = np.asarray(curves)
        out[strat] = {"budgets": b.tolist(),
                      "acc_mean": curves.mean(axis=0).round(4).tolist(),
                      "acc_std": curves.std(axis=0).round(4).tolist(),
                      "labels_to_90": labels_to_target(
                          b, curves.mean(axis=0))}
    return out


def _plot(path, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = {"random": ("random labeling", "#888"),
             "entropy": ("entropy (uncertainty)", "#c86"),
             "entdiv": ("entropy + diversity", "#26a")}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax = axes[0]
    for k, (lab, c) in names.items():
        b = np.asarray(res[k]["budgets"])
        m = np.asarray(res[k]["acc_mean"])
        s = np.asarray(res[k]["acc_std"])
        ax.plot(b, m, "o-", ms=4, color=c, label=lab)
        ax.fill_between(b, m - s, m + s, color=c, alpha=0.15)
    ax.axhline(TARGET_ACC, color="k", ls="--", lw=1,
               label=f"target {TARGET_ACC:.0%}")
    ax.set_xlabel("labels spent")
    ax.set_ylabel("test accuracy (3-class kerf grade)")
    ax.set_title("Accuracy vs label budget (3 seeds)")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    ks = list(names)
    vals = [res[k]["labels_to_90"] or max(res[k]["budgets"]) for k in ks]
    bars = ax.bar([names[k][0] for k in ks], vals,
                  color=[names[k][1] for k in ks])
    for bar, k in zip(bars, ks):
        v = res[k]["labels_to_90"]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{v}" if v else f">{max(res[k]['budgets'])}", ha="center",
                fontsize=10)
    ax.set_ylabel(f"labels needed for {TARGET_ACC:.0%}")
    ax.set_title("The number a manager asks for")
    ax.tick_params(axis="x", labelsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    res = run()
    with open(RESULTS_JSON, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: {"labels_to_90": v["labels_to_90"],
                          "final_acc": v["acc_mean"][-1]}
                      for k, v in res.items()}, indent=2))
    os.makedirs(ASSETS, exist_ok=True)
    for d in (FIGDIR, ASSETS):
        _plot(os.path.join(d, "adc_al.png"), res)
    print("figure ->", FIGDIR, "and", ASSETS)


if __name__ == "__main__":
    main()
