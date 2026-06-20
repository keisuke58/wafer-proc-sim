"""
insitu.listen — process-embedded SHM for dicing ("結晶の聴診器").

Idea (cross-over): treat the wafer *being diced* as a structure under structural
health monitoring. Healthy cutting, micro-chipping and subsurface crack
initiation each radiate distinct acoustic-emission (AE) bursts. We monitor that
stream and flag incipient cracks **before** they become latent field failures —
exactly the Stage-0 novelty-detection step from structure-agnostic SHM, ported
from guided-wave/COPV monitoring to in-process dicing.

The detector is unsupervised: fit a baseline on the first (healthy) windows of a
cut, then score every window by Mahalanobis distance and alarm above a χ²
threshold. No crack labels are needed at run time — the cut teaches its own
baseline. Deeper subsurface damage (from quantum.ProcessSignature) → more, larger
AE bursts → earlier, denser alarms. So the wet, high-damage blade lights up while
dry stealth stays quiet.

Everything here is a transparent **synthetic-AE surrogate** (literature-flavoured
burst shapes, plausible rates); real Hsu-Nielsen-calibrated AE from a DISCO
spindle is required before any number is trusted. The message is the *method*
(self-baselining maha detection on AE) and the *relative* behaviour, not the digits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from quantum.coherence import ProcessSignature, CANONICAL_SIGNATURES


# ── AE event = (time_s, amplitude, is_crack) ──────────────────────────────────
AEEvent = Tuple[float, float, bool]


@dataclass
class AETrace:
    fs: float                       # sample rate [Hz]
    t: np.ndarray                   # time axis [s]
    x: np.ndarray                   # AE waveform [a.u.]
    events: List[AEEvent]           # discrete bursts injected
    catastrophe_s: float | None     # time of the catastrophic chip, if any


def _burst(n: int, fs: float, amp: float, f_res: float, tau_s: float) -> np.ndarray:
    """A single decaying-sinusoid AE burst (Hsu-Nielsen-like)."""
    tt = np.arange(n) / fs
    return amp * np.exp(-tt / tau_s) * np.sin(2 * np.pi * f_res * tt)


def simulate_ae(sig: ProcessSignature, duration_s: float = 0.20,
                fs: float = 2.0e5, seed: int = 0) -> AETrace:
    """Synthesize an AE stream for one dicing pass with the given damage signature.

    Burst rate and amplitude scale with the subsurface-damage depth; a wet, deep
    cut additionally accelerates (grit loading) and ends in a catastrophic chip.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs

    # background grinding noise — louder for deeper/wet cuts
    bg = 0.03 + 0.015 * sig.ssd_um / 10.0 + (0.02 if sig.wet else 0.0)
    x = rng.normal(0.0, bg, n)

    # AE hit rate grows with damage; deep/wet cuts ramp up over the pass (loading)
    rate0 = 60.0 + 90.0 * sig.ssd_um / 10.0            # hits/s at start
    ramp = (2.0 if (sig.wet and sig.ssd_um > 5) else 1.0)
    events: List[AEEvent] = []
    burst_len = int(0.5e-3 * fs)                        # 0.5 ms burst support

    # thin-time Poisson sampling of burst onsets
    n_slots = 200
    for k in range(n_slots):
        frac = k / n_slots
        lam = rate0 * (1.0 + (ramp - 1.0) * frac) * (duration_s / n_slots)
        if rng.random() > lam:
            continue
        i0 = int(frac * n)
        # amplitude: micro-chips small, occasional crack large (heavier tail w/ damage)
        big = rng.random() < (0.05 + 0.20 * sig.ssd_um / 10.0)
        amp = (rng.uniform(0.4, 0.9) if big else rng.uniform(0.08, 0.25)) * (
            1.0 + 0.5 * sig.ssd_um / 10.0)
        f_res = rng.uniform(0.8e5, 2.5e5)               # 80–250 kHz resonance
        seg = _burst(min(burst_len, n - i0), fs, amp, f_res, tau_s=8e-5)
        x[i0:i0 + len(seg)] += seg
        events.append((i0 / fs, amp, bool(big)))

    # catastrophic chip near the end of a deep, wet cut
    catastrophe_s = None
    if sig.wet and sig.ssd_um > 5:
        ic = int(0.88 * n)
        seg = _burst(min(burst_len * 3, n - ic), fs, 1.5, 1.2e5, tau_s=2e-4)
        x[ic:ic + len(seg)] += seg
        catastrophe_s = ic / fs
        events.append((ic / fs, 1.5, True))

    return AETrace(fs=fs, t=t, x=x, events=events, catastrophe_s=catastrophe_s)


# ── windowed AE features ──────────────────────────────────────────────────────
FEATURE_NAMES = ("rms", "peak", "hit_rate", "dom_freq_kHz", "kurtosis")


def _kurtosis(w: np.ndarray) -> float:
    m = w.mean()
    s = w.std() + 1e-12
    return float(np.mean(((w - m) / s) ** 4))


def window_features(trace: AETrace, win_ms: float = 2.0,
                    hit_thresh: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    """Slice the AE trace into windows and extract a feature vector per window.

    Returns (features [n_win, 5], window_start_times [n_win]).
    """
    w = int(win_ms * 1e-3 * trace.fs)
    n_win = len(trace.x) // w
    feats = np.zeros((n_win, len(FEATURE_NAMES)))
    times = np.zeros(n_win)
    for i in range(n_win):
        seg = trace.x[i * w:(i + 1) * w]
        rms = float(np.sqrt(np.mean(seg ** 2)))
        peak = float(np.max(np.abs(seg)))
        hits = int(np.sum(np.abs(seg) > hit_thresh))
        spec = np.abs(np.fft.rfft(seg))
        freqs = np.fft.rfftfreq(len(seg), 1.0 / trace.fs)
        dom = float(freqs[np.argmax(spec)]) / 1e3            # kHz
        feats[i] = (rms, peak, hits / (win_ms * 1e-3), dom, _kurtosis(seg))
        times[i] = i * w / trace.fs
    return feats, times


# ── self-baselining Mahalanobis novelty detector (SHM Stage-0) ────────────────
@dataclass
class MahaDetector:
    """Fit a healthy baseline, score windows by Mahalanobis distance, alarm above
    a threshold taken from the *baseline's own* distance distribution.

    AE features are bursty and non-Gaussian, so a theoretical χ² threshold
    over-alarms. As in guided-wave SHM we instead calibrate the control limit on
    the healthy baseline itself (its max distance² × a safety margin), which
    keeps the false-alarm rate honest under a real, heavy-tailed signal.
    """
    n_baseline: int = 16             # first windows assumed healthy
    margin: float = 1.6              # control limit = margin × max baseline distance²
    mu_: np.ndarray = field(default=None, repr=False)
    Linv_: np.ndarray = field(default=None, repr=False)
    thresh_: float = field(default=0.0)

    def fit(self, feats: np.ndarray) -> "MahaDetector":
        base = feats[: self.n_baseline]
        self.mu_ = base.mean(axis=0)
        cov = np.cov(base, rowvar=False) + 1e-6 * np.eye(feats.shape[1])
        self.Linv_ = np.linalg.inv(np.linalg.cholesky(cov))
        # control limit from the baseline's own distance² distribution
        self.thresh_ = float(np.max(self.distance2(base)) * self.margin)
        return self

    def distance2(self, feats: np.ndarray) -> np.ndarray:
        z = (feats - self.mu_) @ self.Linv_.T
        return np.sum(z ** 2, axis=1)

    def alarms(self, feats: np.ndarray) -> np.ndarray:
        return self.distance2(feats) > self.thresh_


# ── end-to-end monitor + honest metrics ───────────────────────────────────────
@dataclass
class MonitorResult:
    method: str
    times: np.ndarray
    d2: np.ndarray                   # Mahalanobis distance² per window
    alarms: np.ndarray              # bool per window
    cum_energy: np.ndarray          # cumulative AE energy (damage proxy)
    detection_rate: float           # recall on crack-bearing windows
    false_pos_rate: float           # FPR on healthy windows
    lead_time_s: float | None       # first alarm → catastrophe lead time
    catastrophe_s: float | None


def monitor(sig: ProcessSignature, duration_s: float = 0.20,
            fs: float = 2.0e5, seed: int = 0,
            crack_amp: float = 0.4) -> MonitorResult:
    """Run the full listen→feature→detect pipeline and score it on the synthetic
    ground truth (which windows contained a large 'crack' burst)."""
    trace = simulate_ae(sig, duration_s, fs, seed)
    feats, times = window_features(trace)
    det = MahaDetector().fit(feats)
    alarms = det.alarms(feats)
    d2 = det.distance2(feats)

    # cumulative AE energy = ∫ x² (classic SHM damage-accumulation curve)
    w = int(2.0e-3 * fs)
    win_energy = np.array([np.sum(trace.x[i * w:(i + 1) * w] ** 2)
                           for i in range(len(times))])
    cum_energy = np.cumsum(win_energy)

    # ground-truth crack windows: a burst with amp ≥ crack_amp fell in the window
    win_dt = times[1] - times[0] if len(times) > 1 else duration_s
    crack_win = np.zeros(len(times), dtype=bool)
    for (te, amp, _is_crack) in trace.events:
        if amp >= crack_amp:
            j = min(int(te / win_dt), len(times) - 1)
            crack_win[j] = True

    healthy = ~crack_win
    # exclude the baseline region from scoring (it defines 'normal' by construction)
    score = np.ones(len(times), dtype=bool)
    score[: det.n_baseline] = False
    det_rate = (float(np.mean(alarms[crack_win & score]))
                if np.any(crack_win & score) else float("nan"))
    fpr = (float(np.mean(alarms[healthy & score]))
           if np.any(healthy & score) else float("nan"))

    lead = None
    if trace.catastrophe_s is not None and np.any(alarms & score):
        first_alarm = times[np.argmax(alarms & score)]
        lead = max(0.0, trace.catastrophe_s - first_alarm)

    return MonitorResult(
        method=sig.method, times=times, d2=d2, alarms=alarms,
        cum_energy=cum_energy, detection_rate=det_rate, false_pos_rate=fpr,
        lead_time_s=lead, catastrophe_s=trace.catastrophe_s,
    )


def monitor_all(duration_s: float = 0.20, fs: float = 2.0e5, seed: int = 0):
    """Monitor every canonical dicing method; returns {method: MonitorResult}."""
    return {name: monitor(sig, duration_s, fs, seed)
            for name, sig in CANONICAL_SIGNATURES.items()}
