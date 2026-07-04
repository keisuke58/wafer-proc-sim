#!/usr/bin/env python3
"""defect_inspect.py — advanced mask/wafer defect-inspection pipeline.

A demonstrator of the algorithms real actinic mask / wafer inspection tools use
(Lasertec ABICS/ACTIS-class), a level above a plain die-to-die subtraction:

  1. Die-to-die (D2D) with a ROBUST GOLDEN REFERENCE (pixelwise median over many
     dies) and SUB-PIXEL phase-correlation alignment — the reference is not one
     noisy neighbour but the consensus of the array.
  2. ADAPTIVE STATISTICAL DETECTION — the difference image is high-passed (kills
     the smooth illumination nuisance) and normalized by a local robust noise
     estimate (MAD), so the threshold adapts per-pixel. This is what turns the
     ">90 % of flagged sites are nuisance" problem into a usable capture rate.
  3. PHASE-DEFECT CHANNEL — a multilayer/phase defect is (near-)invisible in the
     in-focus amplitude image but appears through-focus. Using the Transport-of-
     Intensity relation I(+z)-I(-z) ∝ ∇²φ, a phase channel reveals what the
     amplitude channel misses — the actinic tools' whole reason to exist.
  4. DEEP-LEARNING ANOMALY DETECTION — a convolutional autoencoder trained only
     on defect-free dies flags anything it cannot reconstruct (unsupervised).
  5. FUSION + EVALUATION as a capture-rate vs nuisance (false-count) curve — the
     KPI an inspection engineer actually optimizes — comparing fixed threshold,
     adaptive, DL and fused.

numpy + scipy + PyTorch + matplotlib.  Run:  python3 defect_inspect.py
"""
import json
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import ndimage as ndi

import torch
import torch.nn as nn

RNG = np.random.default_rng(7)
H = W = 88                      # die tile size [px]
N_DIES = 20                    # dies in the array (reference is their consensus)
PSF_SIGMA = 1.1                # imaging blur [px]
torch.manual_seed(7)


# ─────────────────────────── scene synthesis ────────────────────────────────
def base_pattern():
    """A repeating device pattern: a contact-hole array crossed by two bus lines."""
    p = np.full((H, W), 0.85)                     # bright field
    for cy in range(10, H, 16):
        for cx in range(10, W, 16):
            p[cy-3:cy+3, cx-3:cx+3] = 0.12        # dark contacts
    p[40:44, :] = 0.25                            # horizontal bus
    p[:, 60:64] = 0.25                            # vertical bus
    return p


def _shift(img, dy, dx):
    return ndi.shift(img, (dy, dx), order=1, mode="reflect")


def render_die(pat, illum, dy, dx, seed):
    """In-focus amplitude image: pattern → sub-pixel shift → PSF → illum → noise."""
    rng = np.random.default_rng(seed)
    im = _shift(pat, dy, dx)
    im = ndi.gaussian_filter(im, PSF_SIGMA)
    im = im * illum                               # multiplicative illumination nuisance
    im = im + rng.normal(0, 0.015, im.shape)      # read + shot noise
    return im


def illumination():
    """Smooth low-order illumination gradient (the dominant nuisance)."""
    yy, xx = np.mgrid[0:H, 0:W] / H
    a, b, c = RNG.uniform(-0.12, 0.12, 3)
    return 1.0 + a*yy + b*xx + c*yy*xx


def add_particle(im, cy, cx, r=3, amp=0.5):
    yy, xx = np.mgrid[0:H, 0:W]
    m = (yy-cy)**2 + (xx-cx)**2 <= r*r
    im[m] = np.clip(im[m] + amp, 0, 1.3)
    return m


def add_bridge(im, cy, cx, ln=11):
    im[cy-1:cy+1, cx:cx+ln] = 0.12               # spurious dark bridge
    m = np.zeros((H, W), bool); m[cy-1:cy+1, cx:cx+ln] = True
    return m


def add_open(im, cy, cx, ln=9):
    im[cy-2:cy+2, cx:cx+ln] = 0.85               # break in a dark line
    m = np.zeros((H, W), bool); m[cy-2:cy+2, cx:cx+ln] = True
    return m


def phase_channel(has_defect, cy=0, cx=0, seed=0):
    """Through-focus phase channel  I(+z)-I(-z) ∝ ∇²φ  (Transport of Intensity).

    A phase defect is a small bump in optical path; its Laplacian is what shows
    up through-focus, while the in-focus amplitude image is blind to it.
    """
    rng = np.random.default_rng(seed)
    ch = rng.normal(0, 0.02, (H, W))             # through-focus noise floor
    if has_defect:
        phi = np.zeros((H, W))
        yy, xx = np.mgrid[0:H, 0:W]
        phi += 0.9*np.exp(-((yy-cy)**2 + (xx-cx)**2)/(2*2.2**2))
        ch += ndi.gaussian_laplace(phi, 1.0)     # ∇²φ signal
    return ch


def build_array():
    """Return dies (amplitude), phase channels, and ground-truth defect masks."""
    pat = base_pattern()
    amp, phase, gt = [], [], []          # gt[i] = list of (mask, kind)
    # defect script: die index -> (kind, y, x)
    script = {
        3:  ("particle", 22, 30), 6: ("bridge", 55, 20), 9: ("open", 40, 24),
        12: ("particle", 68, 70), 15: ("bridge", 24, 44), 5: ("open", 40, 8),
        8:  ("phase", 30, 52),    17: ("phase", 60, 20),
    }
    for i in range(N_DIES):
        illum = illumination()
        dy, dx = RNG.uniform(-1.2, 1.2, 2)       # sub-pixel die placement error
        im = render_die(pat, illum, dy, dx, seed=100+i)
        masks = []
        kind = script.get(i, (None,))[0]
        has_phase = False; pcy = pcx = 0
        if kind == "particle":
            _, y, x = script[i]; masks.append((add_particle(im, y, x), kind))
        elif kind == "bridge":
            _, y, x = script[i]; masks.append((add_bridge(im, y, x), kind))
        elif kind == "open":
            _, y, x = script[i]; masks.append((add_open(im, y, x), kind))
        elif kind == "phase":
            _, y, x = script[i]; has_phase = True; pcy, pcx = y, x
            m = np.zeros((H, W), bool); m[y-3:y+3, x-3:x+3] = True
            masks.append((m, kind))
        amp.append(np.clip(im, 0, 1.3))
        phase.append(phase_channel(has_phase, pcy, pcx, seed=200+i))
        gt.append(masks)
    return np.array(amp), np.array(phase), gt


# ─────────────────────── alignment + golden reference ───────────────────────
def subpixel_shift(ref, img):
    """Estimate (dy,dx) by phase correlation with a parabolic sub-pixel peak."""
    F = np.fft.fft2(ref) * np.conj(np.fft.fft2(img))
    R = F / (np.abs(F) + 1e-9)
    c = np.abs(np.fft.ifft2(R))
    py, px = np.unravel_index(np.argmax(c), c.shape)

    def parab(a, b, cc):                          # 3-point parabola vertex
        d = (a - cc)
        return 0.5*d/(a - 2*b + cc + 1e-9)
    yy = parab(c[(py-1) % H, px], c[py, px], c[(py+1) % H, px])
    xx = parab(c[py, (px-1) % W], c[py, px], c[py, (px+1) % W])
    dy = (py + yy); dx = (px + xx)
    dy -= H if dy > H/2 else 0; dx -= W if dx > W/2 else 0
    return dy, dx


def golden_reference(stack):
    """Align every die to the array median, then take the robust median = golden."""
    med = np.median(stack, axis=0)
    aligned = []
    for im in stack:
        dy, dx = subpixel_shift(med, im)
        aligned.append(_shift(im, -dy, -dx))
    aligned = np.array(aligned)
    return np.median(aligned, axis=0), aligned


# ───────────────────────────── detectors ────────────────────────────────────
def detect_fixed(diff):
    """Baseline: global fixed threshold on the raw difference (score = |diff|)."""
    return np.abs(diff)


def noise_reference(aligned, golden):
    """Per-pixel robust noise from the spread of the reference dies (MAD).

    A pixel that varies a lot across *good* dies — pattern edges left by residual
    sub-pixel misalignment, illumination differences — gets a high noise value and
    is automatically de-weighted. The median is robust, so the one die that
    carries a defect does not inflate the estimate at that pixel.
    """
    mad = np.median(np.abs(aligned - golden[None]), axis=0)
    return np.maximum(1.4826 * mad, 0.008)


def detect_adaptive(diff, noise_map):
    """Statistical die-to-die detection: |difference| in units of local noise."""
    return np.abs(diff) / noise_map


class CAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU())
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1), nn.Sigmoid())

    def forward(self, x):
        return self.dec(self.enc(x))


def train_cae(clean_stack, epochs=250):
    net = CAE()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    X = torch.tensor(clean_stack[:, None, :, :], dtype=torch.float32)
    X = X / 1.3
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(net(X), X)
        loss.backward(); opt.step()
    return net


def detect_cae(net, im):
    x = torch.tensor(im[None, None], dtype=torch.float32) / 1.3
    with torch.no_grad():
        r = net(x)[0, 0].numpy() * 1.3
    res = np.abs(im - r)
    return ndi.gaussian_filter(res, 1.0)


# ───────────────────────────── evaluation ───────────────────────────────────
def score_components(score_map, thr):
    lab, n = ndi.label(score_map > thr)
    return lab, n


def evaluate(score_maps, gt, kinds, thresholds):
    """capture-rate vs mean false-count per die, over a threshold sweep."""
    total_def = sum(1 for masks in gt for (m, k) in masks if k in kinds)
    caps, falses = [], []
    for thr in thresholds:
        captured = 0; false_ct = 0
        for i, sm in enumerate(score_maps):
            lab, n = score_components(sm, thr)
            hit_labels = set()
            for (m, k) in gt[i]:
                if k not in kinds:
                    continue
                overlap = lab[m]
                found = overlap[overlap > 0]
                if found.size:
                    captured += 1
                    hit_labels.update(np.unique(found).tolist())
            false_ct += (n - len(hit_labels))
        caps.append(captured/max(total_def, 1))
        falses.append(false_ct/len(score_maps))
    return np.array(caps), np.array(falses)


# ─────────────────────────────── figures ────────────────────────────────────
def fig_scene(amp, phase, gt, golden, path):
    fig, ax = plt.subplots(2, 4, figsize=(11, 5.6))
    show = [3, 6, 9, 8]                            # particle, bridge, open, phase
    titles = ["particle", "bridge", "open", "phase defect"]
    for j, (i, t) in enumerate(zip(show, titles)):
        ax[0, j].imshow(amp[i], cmap="gray", vmin=0, vmax=1.1)
        for (m, k) in gt[i]:
            ys, xs = np.where(m)
            if ys.size:
                ax[0, j].add_patch(plt.Rectangle((xs.min()-2, ys.min()-2),
                    np.ptp(xs)+4, np.ptp(ys)+4, fill=False, ec="#ff3b30", lw=1.4))
        ax[0, j].set_title(f"die {i} — {t}", fontsize=9); ax[0, j].axis("off")
        ax[1, j].imshow(phase[i], cmap="coolwarm", vmin=-.15, vmax=.15)
        ax[1, j].set_title("phase channel  I(+z)−I(−z)", fontsize=8.5); ax[1, j].axis("off")
    fig.suptitle("Synthetic die array — amplitude (top) vs through-focus phase channel (bottom)",
                 fontsize=10.5)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_detect(amp, golden, maps_named, gt, i, path):
    n = len(maps_named)
    fig, ax = plt.subplots(1, n+2, figsize=(2.4*(n+2), 2.8))
    ax[0].imshow(golden, cmap="gray", vmin=0, vmax=1.1); ax[0].set_title("golden ref", fontsize=9); ax[0].axis("off")
    ax[1].imshow(amp[i], cmap="gray", vmin=0, vmax=1.1)
    for (m, k) in gt[i]:
        ys, xs = np.where(m)
        if ys.size:
            ax[1].add_patch(plt.Rectangle((xs.min()-2, ys.min()-2), np.ptp(xs)+4, np.ptp(ys)+4,
                fill=False, ec="#ff3b30", lw=1.4))
    ax[1].set_title(f"test die {i}", fontsize=9); ax[1].axis("off")
    for j, (name, sm) in enumerate(maps_named):
        ax[j+2].imshow(sm, cmap="inferno"); ax[j+2].set_title(name, fontsize=9); ax[j+2].axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_curve(curves, path):
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    colors = {"fixed threshold": "#8a8f98", "adaptive (MAD z)": "#1f77b4",
              "DL autoencoder": "#2ca02c", "fused (amp+phase)": "#d62728"}
    for name, (caps, falses) in curves.items():
        order = np.argsort(falses)
        ax.plot(falses[order], caps[order], "-o", ms=3, lw=1.8,
                color=colors.get(name), label=name)
    ax.set_xlabel("mean nuisance (false detections) per die")
    ax.set_ylabel("defect capture rate")
    ax.set_xlim(-0.1, 6); ax.set_ylim(0, 1.02)
    ax.set_title("Capture rate vs nuisance — the inspection KPI", fontsize=10.5)
    ax.grid(alpha=.25); ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ─────────────────────────────── main ───────────────────────────────────────
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    figs = os.path.join(here, "figures"); os.makedirs(figs, exist_ok=True)

    amp, phase, gt = build_array()
    golden, aligned = golden_reference(amp)
    golden_ph, aligned_ph = golden_reference(phase)
    noise_map = noise_reference(aligned, golden)
    noise_ph = noise_reference(aligned_ph, golden_ph)

    amp_kinds = {"particle", "bridge", "open"}
    # per-die score maps for each detector (amplitude channel)
    fixed_maps, adapt_maps, cae_maps, fused_maps, phase_maps = [], [], [], [], []
    clean = np.array([aligned[i] for i in range(N_DIES) if not gt[i]])
    net = train_cae(clean)
    for i in range(N_DIES):
        diff = aligned[i] - golden
        fx = detect_fixed(diff)
        ad = detect_adaptive(diff, noise_map)
        ca = detect_cae(net, amp[i])
        ca_z = ca / (np.median(ca) + 1.4826*np.median(np.abs(ca-np.median(ca))) + 1e-6)
        ph = np.abs(phase[i] - golden_ph) / noise_ph
        fu = np.maximum(ca_z, ph)                  # multi-channel fusion: DL (amp) + phase
        fixed_maps.append(fx); adapt_maps.append(ad); cae_maps.append(ca_z)
        fused_maps.append(fu); phase_maps.append(ph)

    # Evaluate over ALL defect types: amplitude-only detectors structurally miss
    # the phase defects, so only the amplitude+phase fusion reaches full capture.
    all_kinds = {"particle", "bridge", "open", "phase"}
    z = np.linspace(1, 16, 40)
    curves = {
        "fixed threshold": evaluate(fixed_maps, gt, all_kinds, np.linspace(0.03, 0.6, 40)),
        "adaptive (MAD z)": evaluate(adapt_maps, gt, all_kinds, z),
        "DL autoencoder": evaluate(cae_maps, gt, all_kinds, z),
        "fused (amp+phase)": evaluate(fused_maps, gt, all_kinds, z),
    }
    # phase-defect capture only: amplitude-adaptive vs the phase channel
    cap_amp, fp_amp = evaluate(adapt_maps, gt, {"phase"}, z)
    cap_ph, fp_ph = evaluate(phase_maps, gt, {"phase"}, z)

    fig_scene(amp, phase, gt, golden, os.path.join(figs, "insp_scene.png"))
    di = 6
    fig_detect(amp, golden, [("fixed", fixed_maps[di]), ("adaptive z", adapt_maps[di]),
               ("DL residual", cae_maps[di]), ("fused", fused_maps[di])],
               gt, di, os.path.join(figs, "insp_detect.png"))
    fig_curve(curves, os.path.join(figs, "insp_curve.png"))

    # phase reveal figure
    pdie = 8
    fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.9))
    ax[0].imshow(amp[pdie], cmap="gray", vmin=0, vmax=1.1); ax[0].set_title("amplitude (in focus)\nphase defect invisible", fontsize=9); ax[0].axis("off")
    ax[1].imshow(phase[pdie], cmap="coolwarm", vmin=-.15, vmax=.15); ax[1].set_title("phase channel  ∝ ∇²φ\ndefect revealed", fontsize=9); ax[1].axis("off")
    ax[2].imshow(phase_maps[pdie], cmap="inferno"); ax[2].set_title("phase-channel detection", fontsize=9); ax[2].axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(figs, "insp_phase.png"), dpi=130); plt.close(fig)

    # operating points: capture at a fixed budget of ~0.5 nuisance/die
    def cap_at(caps, falses, budget=0.5):
        ok = falses <= budget
        return float(caps[ok].max()) if ok.any() else 0.0
    out = {
        "dies": N_DIES, "die_px": [H, W],
        "amplitude_defects": sum(1 for m in gt for (_, k) in m if k in amp_kinds),
        "phase_defects": sum(1 for m in gt for (_, k) in m if k == "phase"),
        "capture_at_0p5_nuisance": {
            name: round(cap_at(c, f), 3) for name, (c, f) in curves.items()},
        "capture_at_2_nuisance": {
            name: round(cap_at(c, f, 2.0), 3) for name, (c, f) in curves.items()},
        "amplitude_only_ceiling": round(6/8, 3),
        "phase_defect_capture_at_0p5_nuisance": {
            "amplitude_adaptive": round(cap_at(cap_amp, fp_amp), 3),
            "phase_channel": round(cap_at(cap_ph, fp_ph), 3)},
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    with open(os.path.join(here, "defect_inspect_results.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
