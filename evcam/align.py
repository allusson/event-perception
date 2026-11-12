"""Aligning a phone slow-motion clip with a DAVIS recording of the same scene (notebooks 06, 10).

Time: `sync_by_event_rate` simulates events from the phone frames and cross-correlates the two
event rates, so both sides are the same quantity in the same units. The convention everywhere is
t_davis = t_phone + offset, both in seconds on each clip's own clock (0 at its first frame / event).

Space: `map_by_keypoints` (similarity from matched body joints) and `map_by_features`
(ORB + RANSAC homography between a phone frame and an APS frame), both phone pixels -> DAVIS pixels.
"""

import dataclasses

import cv2
import numpy as np

from evcam.simulate import simulate_events

MIN_SCORE = 0.5            # reliable: peak correlation at least this
MIN_PEAK_RATIO = 1.2       # and at least this times the highest other peak
OTHER_PEAK_S = 0.5         # "other" peaks are at least this far from the best lag


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SyncResult:
    """Time alignment t_davis = t_phone + offset_s.

    offset_s : seconds
    score : peak normalized cross-correlation of the two event rates, in [0, 1]
    peak_ratio : peak / highest other peak at least OTHER_PEAK_S away (inf if there is none)
    reliable : score >= MIN_SCORE and peak_ratio >= MIN_PEAK_RATIO
    source : "auto" (cross-correlation) or "manual" (from a pair of times)
    curves : dict with "t_davis", "rate_davis", "t_phone", "rate_phone" (bin centres in s on each
        clip's own clock, events / s) and "lag_s", "ncc" (the correlation over the searched offsets)
    """
    offset_s: float
    score: float
    peak_ratio: float
    reliable: bool
    source: str = "auto"
    curves: dict = dataclasses.field(default_factory=dict, repr=False)


def event_rate(t_us, t_end_us=None, bin_ms=10):
    """Event rate in bins of `bin_ms` from 0 to `t_end_us` (default: last event).

    Returns (t_s, rate): bin centres in seconds and events per second.
    """
    t_end_us = int(t_us[-1]) + 1 if t_end_us is None else int(t_end_us)
    edges = np.arange(0, t_end_us + bin_ms * 1000, bin_ms * 1000)
    counts = np.histogram(t_us, edges)[0]
    return (edges[:-1] + edges[1:]) / 2e6, counts / (bin_ms / 1e3)


def _ncc_valid(long, short):
    """Pearson correlation of `short` with every window of `long` of the same length."""
    n = len(short)
    s = (short - short.mean()) / (short.std() + 1e-12)
    num = np.correlate(long, s, mode="valid") / n
    c1 = np.concatenate([[0.0], np.cumsum(long)])
    c2 = np.concatenate([[0.0], np.cumsum(long ** 2)])
    mean = (c1[n:] - c1[:-n]) / n
    var = np.maximum((c2[n:] - c2[:-n]) / n - mean ** 2, 0.0)
    return np.where(var > 0, num / np.sqrt(var + 1e-12), 0.0)


def _peak_ratio(ncc, k, bin_s):
    far = np.abs(np.arange(len(ncc)) - k) * bin_s >= OTHER_PEAK_S
    if not far.any():
        return np.inf
    other = ncc[far].max()
    return np.inf if other <= 0 else float(ncc[k] / other)


def sync_by_event_rate(davis_events, phone_clip, bin_ms=10, c=0.2, max_side=128):
    """Time offset between a phone clip and a DAVIS recording from their event rates.

    The phone frames are shrunk to `max_side` pixels on the long side (only the timing matters)
    and turned into events with `simulate_events` at threshold `c`. Both event rates (events / s,
    `bin_ms` bins, each on its own clock) are compared with the normalized cross-correlation:
    each is z-scored over the overlap, so the different thresholds and resolutions do not matter.
    The offset is the lag of the highest correlation over all lags where the shorter signal fits
    inside the longer one, refined between bins with a parabola through the peak.

    Parameters
    ----------
    davis_events : DAVIS events (EVENT_DTYPE), t in microseconds from the start of the recording
    phone_clip : Clip with frames and frame_t (microseconds from the first frame)

    Returns
    -------
    SyncResult with t_davis = t_phone + offset_s
    """
    H, W = phone_clip.shape
    s = min(1.0, max_side / max(H, W))
    frames = phone_clip.frames
    if s < 1:
        size = (max(int(round(W * s)), 2), max(int(round(H * s)), 2))
        frames = np.stack([cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames])
    small = dataclasses.replace(phone_clip, frames=frames, shape=frames.shape[1:], events=None)
    sim_t = simulate_events(small, c=c).events["t"]
    t_p, r_p = event_rate(sim_t, phone_clip.frame_t[-1], bin_ms)
    t_d, r_d = event_rate(davis_events["t"], None, bin_ms)
    bin_s = bin_ms / 1e3

    if len(r_p) <= len(r_d):
        ncc = _ncc_valid(r_d, r_p)
        lag_s = np.arange(len(ncc)) * bin_s                    # davis bin k <-> phone bin 0
    else:
        ncc = _ncc_valid(r_p, r_d)[::-1]                        # phone bin k <-> davis bin 0: offset -k
        lag_s = -np.arange(len(ncc))[::-1] * bin_s
    k = int(np.argmax(ncc))
    frac = 0.0
    if 0 < k < len(ncc) - 1:
        a, b, cc = ncc[k - 1:k + 2]
        den = a - 2 * b + cc
        frac = float(np.clip(0.5 * (a - cc) / den, -0.5, 0.5)) if den < 0 else 0.0
    offset = float(lag_s[k] + frac * bin_s)
    score = float(np.clip(ncc[k], 0, 1))
    ratio = _peak_ratio(ncc, k, bin_s)
    curves = {"t_davis": t_d, "rate_davis": r_d, "t_phone": t_p, "rate_phone": r_p, "lag_s": lag_s, "ncc": ncc,
              "bin_ms": bin_ms, "c": c}
    return SyncResult(offset, score, ratio, score >= MIN_SCORE and ratio >= MIN_PEAK_RATIO, "auto", curves)


def sync_from_pair(t_phone_s, t_davis_s):
    """Offset from one moment seen in both clips: t_davis = t_phone + offset."""
    return float(t_davis_s) - float(t_phone_s)


def check_sync(sync, name="SYNC"):
    """Validate a manual sync setting: None, or (t_phone_s, t_davis_s) in seconds. Returns it."""
    ok = sync is None or (isinstance(sync, (tuple, list)) and len(sync) == 2
                          and all(isinstance(v, (int, float)) and np.isfinite(v) and v >= 0 for v in sync))
    if not ok:
        raise ValueError(f"{name} must be None or (t_phone_s, t_davis_s) in seconds, e.g. (1.40, 12.35); got {sync!r}")
    return sync


def resolve_sync(auto, sync=None):
    """The SyncResult to use: `auto`, or a manual one from SYNC = (t_phone_s, t_davis_s) (curves kept)."""
    if check_sync(sync) is None:
        return auto
    return dataclasses.replace(auto, offset_s=sync_from_pair(*sync), source="manual")


def motion_start(t_s, rate, t_from_s, n_sigma=5.0, min_factor=1.5, baseline_s=1.0, smooth_s=0.05):
    """First time >= t_from_s where the event rate rises clearly above its still level.

    The rate is smoothed over `smooth_s`. At each bin, the still level is the median m of the
    preceding `baseline_s` and its spread the robust standard deviation s = 1.4826 x MAD; the rate
    rises clearly when it exceeds both m + n_sigma * s and min_factor * m. (A plain multiple of the
    median does not work on the DAVIS: its noise floor keeps the median of a still scene high.)

    t_s, rate : from `event_rate`. Returns the start of that bin in seconds, or None.
    """
    bin_s = t_s[1] - t_s[0]
    ns = max(int(round(smooth_s / bin_s)), 1)
    r = np.convolve(rate, np.ones(ns) / ns, mode="same")
    nb = max(int(round(baseline_s / bin_s)), 2)
    for k in np.flatnonzero(t_s >= t_from_s):
        if k < nb:
            continue
        base = r[k - nb:k]
        m = np.median(base)
        s = 1.4826 * np.median(np.abs(base - m))
        if r[k] > max(m + n_sigma * s, min_factor * m) and r[k] > 0:
            return float(t_s[k] - bin_s / 2)
    return None


def _z(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def plot_alignment(res, figsize=(13, 5.2)):
    """Figure 1: real and simulated event rate, each over its whole clip on its own clock, minor ticks
    every 0.5 s (for a manual SYNC). Figure 2: both z-scored on the DAVIS clock after alignment."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    cv = res.curves
    fig, ax = plt.subplots(2, 1, figsize=figsize, sharey=False)
    ax[0].plot(cv["t_davis"], cv["rate_davis"] / 1e6, lw=0.7, color="tab:blue")
    ax[0].set_xlabel("DAVIS time (s)"); ax[0].set_title(f"real: DAVIS event rate, {cv['bin_ms']} ms bins")
    ax[1].plot(cv["t_phone"], cv["rate_phone"] / 1e6, lw=0.7, color="tab:orange")
    ax[1].set_xlabel("phone time (s)")
    ax[1].set_title(f"simulated from the phone video (c = {cv['c']}, reduced resolution), {cv['bin_ms']} ms bins")
    for a in ax:
        a.set_ylabel("M events / s"); a.xaxis.set_minor_locator(MultipleLocator(0.5)); a.grid(which="both", lw=0.3)
    fig.tight_layout(); plt.show()

    fig2, a = plt.subplots(figsize=(figsize[0], 2.8))
    t0, t1 = cv["t_phone"][0] + res.offset_s, cv["t_phone"][-1] + res.offset_s
    sel = (cv["t_davis"] >= t0) & (cv["t_davis"] <= t1)
    a.plot(cv["t_davis"][sel], _z(cv["rate_davis"][sel]), lw=0.8, label="real (DAVIS), z-scored")
    a.plot(cv["t_phone"] + res.offset_s, _z(cv["rate_phone"]), lw=0.8, alpha=0.8, label="simulated (phone), z-scored")
    a.set_xlabel("DAVIS time (s)"); a.set_ylabel("rate, z-score (no unit)")
    a.set_title(f"after alignment: t_davis = t_phone + {res.offset_s:.3f} s ({res.source})")
    a.xaxis.set_minor_locator(MultipleLocator(0.5)); a.grid(which="both", lw=0.3); a.legend(fontsize=8)
    fig2.tight_layout(); plt.show()
    return fig, fig2


def describe(res):
    """One line: offset, score, peak ratio, reliability, source."""
    rel = "reliable" if res.reliable else f"NOT reliable (needs score >= {MIN_SCORE}, peak ratio >= {MIN_PEAK_RATIO})"
    return (f"offset t_davis - t_phone = {res.offset_s:.3f} s ({res.source}); automatic estimate: score "
            f"{res.score:.2f}, peak ratio {res.peak_ratio:.2f}, {rel}")


# ---------------------------------------------------------------------------
# Space
# ---------------------------------------------------------------------------

def activity_box(img, frac=0.5, pct=99, margin=0.1, keep=0.96):
    """Box (x0, y0, x1, y1) in pixels around where `img` (e.g. an event count image) is busy.

    Busy pixels are those above frac x the pct-th percentile of `img`. The box spans the central
    `keep` share of their values along x and along y (so isolated noise pixels do not stretch it),
    grown by `margin` x its size on each side and clipped to the image. None if nothing is busy.
    """
    H, W = img.shape
    thr = frac * np.percentile(img, pct)
    ys, xs = np.nonzero(img > thr) if thr > 0 else np.nonzero(img > 0)
    if len(xs) == 0:
        return None
    w = img[ys, xs].astype(float)
    q = [(1 - keep) / 2, (1 + keep) / 2]
    wq = lambda v: np.interp(q, np.cumsum(w[np.argsort(v)]) / w.sum(), np.sort(v))
    (x0, x1), (y0, y1) = wq(xs), wq(ys)
    return grow_box((x0, y0, x1 + 1, y1 + 1), margin, (W, H))


def grow_box(box, margin, wh):
    """Grow (x0, y0, x1, y1) by margin x its size on each side, clipped to the image (W, H); ints."""
    x0, y0, x1, y1 = box
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    W, H = wh
    return (int(max(x0 - mx, 0)), int(max(y0 - my, 0)), int(min(np.ceil(x1 + mx), W)), int(min(np.ceil(y1 + my), H)))


def _similarity(p, q):
    """Least-squares similarity p -> q (Umeyama) as a 2x3 matrix; p, q (N, 2) without NaN."""
    mp, mq = p.mean(0), q.mean(0)
    P, Q = p - mp, q - mq
    U, S, Vt = np.linalg.svd(Q.T @ P / len(p))
    D = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    scale = np.trace(np.diag(S) @ D) / (P ** 2).sum(1).mean()
    return np.hstack([scale * R, (mq - scale * R @ mp)[:, None]])


def _apply(A, p):
    return p @ A[:, :2].T + A[:, 2]


# left / right joint pairs of the 13-joint format (shoulders, elbows, hips, hands, knees, feet)
LR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]


def _swap_lr(kp):
    out = kp.copy()
    for a, b in LR_PAIRS:
        out[..., [a, b], :] = kp[..., [b, a], :]
    return out


def map_by_keypoints(kp_phone, kp_davis, min_points=4):
    """Least-squares similarity (scale, rotation, translation) phone pixels -> DAVIS pixels from matched joints.

    kp_phone, kp_davis : (frames, 13, 2) joints in each camera's own pixels (format of
        `evcam.controlevents`; a single (13, 2) frame also works), NaN = missing; a joint is used
        only if present in both
    OpenPose sometimes swaps left and right, more often on the grayscale APS frames, so the fit
    starts from points that do not depend on the labelling (head and the midpoint of each
    left / right pair), then keeps per frame whichever DAVIS labelling (as given or swapped) fits
    better and refits on all joints.

    Returns (A, rms): A the 2x3 matrix (x_davis = A @ [x_phone, 1]) and the RMS residual in DAVIS
    pixels, or None with fewer than `min_points` matched joints.
    """
    p = np.asarray(kp_phone, float).reshape(-1, 13, 2)
    q = np.asarray(kp_davis, float).reshape(-1, 13, 2)

    def fit(p, q):
        pp, qq = p.reshape(-1, 2), q.reshape(-1, 2)
        ok = np.isfinite(pp).all(1) & np.isfinite(qq).all(1)
        return (_similarity(pp[ok], qq[ok]), ok.sum()) if ok.sum() >= max(min_points, 2) else (None, ok.sum())

    mid = lambda k: np.concatenate([k[:, :1], (k[:, [a for a, _ in LR_PAIRS]] + k[:, [b for _, b in LR_PAIRS]]) / 2], 1)
    A, _ = fit(mid(p), mid(q))
    if A is None:
        A, _ = fit(p, q)
        if A is None:
            return None
    for _ in range(3):
        err = lambda qq: np.nanmean(np.where(np.isfinite(qq), (_apply(A, p) - qq) ** 2, np.nan).sum(2), 1)
        swap = err(_swap_lr(q)) < err(q)
        q = np.where(swap[:, None, None], _swap_lr(q), q)
        A, n = fit(p, q)
        if A is None:
            return None
    pp, qq = p.reshape(-1, 2), q.reshape(-1, 2)
    ok = np.isfinite(pp).all(1) & np.isfinite(qq).all(1)
    rms = float(np.sqrt(((_apply(A, pp[ok]) - qq[ok]) ** 2).sum(1).mean()))
    return A, rms


def _gray_u8(img):
    img = np.asarray(img)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.dtype == np.uint8 else img.mean(2)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img


def map_by_features(phone_frame, aps_frame, min_inliers=15):
    """Homography phone pixels -> DAVIS pixels from ORB features and RANSAC.

    phone_frame : phone image (grayscale or RGB; uint8, or float in [0, 1]) in its own pixels
    aps_frame : DAVIS APS frame (H, W), uint8 or float in [0, 1]
    The phone frame is resized toward the APS resolution first (same height) and both are
    contrast-equalized, so the features have a similar scale. Returns (Hm, n_inliers) with Hm the
    3x3 homography on the phone frame's own pixels, or None with fewer than `min_inliers` inliers.
    """
    a, b = _gray_u8(phone_frame), _gray_u8(aps_frame)
    s = b.shape[0] / a.shape[0]
    a = cv2.resize(a, (max(int(round(a.shape[1] * s)), 1), b.shape[0]), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    a, b = clahe.apply(a), clahe.apply(b)
    orb = cv2.ORB_create(4000, fastThreshold=5)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < min_inliers or len(kb) < min_inliers:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m for m, *rest in pairs if rest and m.distance < 0.8 * rest[0].distance]
    if len(good) < min_inliers:
        return None
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    Hm, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if Hm is None or mask.sum() < min_inliers or not _near_similarity(Hm, a.shape):
        return None
    return Hm @ np.diag([s, s, 1.0]), int(mask.sum())


def _near_similarity(Hm, shape, max_perspective=1.5, max_stretch=1.6):
    """The cameras sit side by side, so the true mapping is close to a similarity. Reject a
    homography whose perspective changes the scale by more than `max_perspective` across the frame,
    whose local stretch is anisotropic by more than `max_stretch`, or whose orientation flips
    (repetitive textures such as blinds give such wrong fits)."""
    h, w = shape
    corners = np.float64([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1], [w / 2, h / 2, 1]])
    z = corners @ Hm[2]
    if (z <= 0).any() or z.max() / z.min() > max_perspective:
        return False
    J = (Hm[:2, :2] * z[4] - np.outer(Hm[:2] @ corners[4], Hm[2, :2])) / z[4] ** 2
    sv = np.linalg.svd(J, compute_uv=False)
    return np.linalg.det(J) > 0 and sv[0] / sv[1] <= max_stretch
