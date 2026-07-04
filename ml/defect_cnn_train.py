#!/usr/bin/env python3
"""Train the wafer-defect CNN and export weights for the C++ inference engine.

A compact conv net (conv 8x5x5 -> ReLU -> maxpool 4x4 -> dense 128->4 -> softmax)
trained from scratch in NumPy on synthetic 20x20 defect patches. Exports:

  results/defect_cnn_weights.bin   float32: conv1_w[8,1,5,5], conv1_b[8],
                                            fc_w[4,128], fc_b[4]
  results/defect_cnn_testset.bin   int32 N,H,W ; N*H*W float32 (0..255) ; N int32 labels
  results/defect_cnn_samples.png   montage of example patches per class

Classes: 0 good, 1 chipping, 2 crack, 3 contamination.  Inference is done in C++
(vision/cpp/src/cnn.cpp); this file only trains and serializes.
"""
import os
import struct

import numpy as np

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "..", "results")
IN = 20            # patch size
K = 5              # conv kernel
C1 = 8             # conv filters
CONV_OUT = IN - K + 1     # 16
POOL = 4
POOL_OUT = CONV_OUT // POOL  # 4
FLAT = C1 * POOL_OUT * POOL_OUT  # 128
NC = 4
CLASS_NAMES = ["good", "chipping", "crack", "contamination"]


# ── Synthetic defect patches ─────────────────────────────────────────────────

def make_patch(cls: int, rng: np.random.Generator) -> np.ndarray:
    """One 20x20 grayscale defect patch (values ~0..255)."""
    img = 130.0 + rng.normal(0, 6, (IN, IN))
    if cls == 0:                                   # good: mild texture only
        pass
    elif cls == 1:                                 # chipping: notch on an edge
        bright = rng.random() < 0.5
        val = 215.0 if bright else 45.0
        depth = rng.integers(5, 9)
        length = rng.integers(6, 12)
        edge = rng.integers(0, 4)
        s = rng.integers(0, IN - length)
        for d in range(depth):
            for t in range(length - d):            # tapered notch
                if edge == 0:   img[d, s + t] = val
                elif edge == 1: img[IN - 1 - d, s + t] = val
                elif edge == 2: img[s + t, d] = val
                else:           img[s + t, IN - 1 - d] = val
    elif cls == 2:                                 # crack: thin dark line
        val = 40.0
        y0, x0 = rng.integers(2, IN - 2), rng.integers(2, IN - 2)
        ang = rng.uniform(0, np.pi)
        dx, dy = np.cos(ang), np.sin(ang)
        for s in range(-14, 15):
            x = int(round(x0 + s * dx)); y = int(round(y0 + s * dy))
            if 0 <= x < IN and 0 <= y < IN:
                img[y, x] = val
                if 0 <= x + 1 < IN:
                    img[y, x + 1] = 0.6 * val + 0.4 * img[y, x + 1]
    else:                                          # contamination: dark specks
        for _ in range(rng.integers(6, 12)):
            y, x = rng.integers(1, IN - 1), rng.integers(1, IN - 1)
            r = rng.integers(1, 3)
            img[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = rng.uniform(30, 70)
    return np.clip(img, 0, 255)


def make_dataset(n_per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(NC):
        for _ in range(n_per_class):
            X.append(make_patch(c, rng))
            y.append(c)
    X = np.stack(X).astype(np.float64)
    y = np.array(y, dtype=np.int64)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# ── Layers (NumPy, manual backprop) ──────────────────────────────────────────

def im2col(x, k):
    """x: [N,H,W] -> cols: [N, k*k, OH*OW] (single input channel)."""
    n, h, w = x.shape
    oh, ow = h - k + 1, w - k + 1
    cols = np.empty((n, k * k, oh * ow), dtype=x.dtype)
    for ky in range(k):
        for kx in range(k):
            patch = x[:, ky:ky + oh, kx:kx + ow]         # [N,oh,ow]
            cols[:, ky * k + kx, :] = patch.reshape(n, -1)
    return cols, oh, ow


def forward(params, x):
    """Return logits and a cache for backprop. x: [N,20,20] in [0,1]."""
    W1, b1, Wf, bf = params
    n = x.shape[0]
    cols, oh, ow = im2col(x, K)                          # [N,25,256]
    W1f = W1.reshape(C1, K * K)                          # [8,25]
    convf = np.einsum("ok,nkp->nop", W1f, cols) + b1[None, :, None]  # [N,8,256]
    conv = convf.reshape(n, C1, oh, ow)
    relu = np.maximum(conv, 0.0)
    # max-pool 4x4
    rp = relu.reshape(n, C1, POOL_OUT, POOL, POOL_OUT, POOL)
    pooled = rp.max(axis=(3, 5))                         # [N,8,4,4]
    # argmax mask for backward
    flat_pool = rp.transpose(0, 1, 2, 4, 3, 5).reshape(n, C1, POOL_OUT, POOL_OUT, POOL * POOL)
    amax = flat_pool.argmax(axis=-1)
    flat = pooled.reshape(n, FLAT)
    logits = flat @ Wf.T + bf                            # [N,4]
    cache = (x, cols, conv, relu, pooled, amax, flat, oh, ow)
    return logits, cache


def softmax_ce(logits, y):
    z = logits - logits.max(axis=1, keepdims=True)
    ez = np.exp(z)
    probs = ez / ez.sum(axis=1, keepdims=True)
    n = len(y)
    loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()
    return loss, probs


def backward(params, cache, probs, y):
    W1, b1, Wf, bf = params
    x, cols, conv, relu, pooled, amax, flat, oh, ow = cache
    n = len(y)
    dlogits = probs.copy()
    dlogits[np.arange(n), y] -= 1.0
    dlogits /= n
    dWf = dlogits.T @ flat                               # [4,128]
    dbf = dlogits.sum(axis=0)
    dflat = dlogits @ Wf                                 # [N,128]
    dpool = dflat.reshape(n, C1, POOL_OUT, POOL_OUT)
    # route pool grad back to the argmax position within each 4x4 window
    drelu = np.zeros_like(relu)
    for qy in range(POOL):
        for qx in range(POOL):
            sel = (amax == (qy * POOL + qx))
            ys = (np.arange(POOL_OUT)[:, None] * POOL + qy)
            xs = (np.arange(POOL_OUT)[None, :] * POOL + qx)
            contrib = dpool * sel
            drelu[:, :, ys[:, 0][:, None].repeat(POOL_OUT, 1), xs[0][None, :].repeat(POOL_OUT, 0)] += contrib
    dconv = drelu * (conv > 0.0)                         # [N,8,16,16]
    dconvf = dconv.reshape(n, C1, oh * ow)
    dW1 = np.einsum("nop,nkp->ok", dconvf, cols)         # [8,25]
    db1 = dconvf.sum(axis=(0, 2))
    return (dW1.reshape(C1, 1, K, K), db1,
            dWf.reshape(NC, FLAT), dbf)


# ── Train / export ───────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS, exist_ok=True)
    Xtr, ytr = make_dataset(400, seed=1)
    Xte, yte = make_dataset(120, seed=2)
    xtr = (Xtr / 255.0)
    xte = (Xte / 255.0)

    rng = np.random.default_rng(0)
    W1 = rng.normal(0, np.sqrt(2.0 / (K * K)), (C1, K, K))
    b1 = np.zeros(C1)
    Wf = rng.normal(0, np.sqrt(2.0 / FLAT), (NC, FLAT))
    bf = np.zeros(NC)

    def as_flat_params():
        return (W1, b1, Wf, bf)

    lr, epochs, bs = 0.15, 60, 64
    n = len(ytr)
    for ep in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            logits, cache = forward(as_flat_params(), xtr[b])
            _, probs = softmax_ce(logits, ytr[b])
            dW1, db1, dWf, dbf = backward(as_flat_params(), cache, probs, ytr[b])
            W1 -= lr * dW1.reshape(C1, K, K)
            b1 -= lr * db1
            Wf -= lr * dWf
            bf -= lr * dbf
        if (ep + 1) % 15 == 0 or ep == 0:
            lg, _ = forward(as_flat_params(), xte)
            acc = (lg.argmax(1) == yte).mean()
            print(f"epoch {ep+1:3d}  test acc = {acc*100:.1f}%")

    lg, _ = forward(as_flat_params(), xte)
    acc = (lg.argmax(1) == yte).mean()
    print(f"[final] test accuracy = {acc*100:.2f}%  ({len(yte)} patches)")

    # Export weights (conv1_w[8,1,5,5], conv1_b[8], fc_w[4,128], fc_b[4]).
    wpath = os.path.join(RESULTS, "defect_cnn_weights.bin")
    with open(wpath, "wb") as f:
        f.write(W1.reshape(C1, 1, K, K).astype("<f4").tobytes())
        f.write(b1.astype("<f4").tobytes())
        f.write(Wf.reshape(NC, FLAT).astype("<f4").tobytes())
        f.write(bf.astype("<f4").tobytes())
    print(f"[✓] weights -> {wpath}")

    # Export a labelled test set for the C++ integration test.
    tpath = os.path.join(RESULTS, "defect_cnn_testset.bin")
    with open(tpath, "wb") as f:
        f.write(struct.pack("<iii", len(yte), IN, IN))
        f.write(Xte.astype("<f4").tobytes())            # pixel values 0..255
        f.write(yte.astype("<i4").tobytes())
    print(f"[✓] testset -> {tpath}")

    # Montage of example patches for the dashboard/gallery.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rng2 = np.random.default_rng(7)
        fig, axes = plt.subplots(4, 6, figsize=(7.2, 5.0))
        for c in range(NC):
            for j in range(6):
                ax = axes[c, j]
                ax.imshow(make_patch(c, rng2), cmap="gray", vmin=0, vmax=255)
                ax.set_xticks([]); ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel(CLASS_NAMES[c], fontsize=11)
        fig.suptitle("Wafer-defect patches — CNN training classes "
                     f"(C++ inference test acc {acc*100:.1f}%)", fontsize=12)
        plt.tight_layout()
        mpath = os.path.join(RESULTS, "defect_cnn_samples.png")
        plt.savefig(mpath, dpi=150, bbox_inches="tight")
        print(f"[✓] samples -> {mpath}")
    except Exception as e:            # figure is optional
        print(f"[!] montage skipped: {e}")


if __name__ == "__main__":
    main()
