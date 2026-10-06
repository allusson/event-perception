"""Event-based deblurring: EDI (model-based, no learning). CMax + deconvolution and EFNet
inference are added further down.

All functions take the blurry frame as a float (H, W) image in [0, 1] together with the
frame's exposure window [t_begin, t_end) in microseconds, on the same clock as the events.
`evcam.io.load_aedat4` returns both for DAVIS recordings (`exposure_begin_t`,
`exposure_end_t`).
"""

import warnings

import cv2
import numpy as np

# Width of one time bin when `n_bins` is left to default (microseconds), see `_default_n_bins`.
EDI_BIN_US = 500
GOLDEN = (np.sqrt(5) - 1) / 2


# ---------------------------------------------------------------------------
# EDI: Event-based Double Integral (Pan et al., CVPR 2019)
# ---------------------------------------------------------------------------
#
# Model. An event fires when log intensity changes by the contrast threshold c, so the
# latent sharp image at time t relates to the one at t_ref by
#     L(t) = L(t_ref) * exp(c * E(t)),
# with E(t) the signed event count between t_ref and t. The blurry frame is the average of
# L(t) over the exposure, hence
#     B = L(t_ref) * mean_t exp(c * E(t))   =>   L(t_ref) = B / divisor.

def _default_n_bins(t_begin, t_end):
    """Number of time bins so that one bin is about EDI_BIN_US (0.5 ms) wide.

    E(t) is stored exactly at the bin edges; binning only loses *when inside a bin* an
    event fired. The trapezoid rule then treats each step in exp(c E) as a ramp across its
    bin, which is exact for an event at the bin centre and off by at most half a bin
    otherwise. So the relative error of the divisor is about (c / 2) * bin / exposure per
    event, with random sign: 0.5 ms bins on a 33 ms DAVIS exposure give <0.3% per event at
    c = 0.3, well below sensor noise and the error from assuming one global c. Finer bins
    cost memory linearly ((n_bins + 1) * H * W floats) for no visible gain; bins much
    coarser than 1 ms start to smear fast edges. Clamped to [16, 512] so very short
    exposures still get a usable integral and very long ones stay within memory.
    """
    return int(np.clip(np.ceil((t_end - t_begin) / EDI_BIN_US), 16, 512))


def event_integral(events, t_begin, t_end, t_ref, shape, n_bins=None):
    """Per-pixel signed event count E(t) between t_ref and each bin edge of the exposure.

    This is the part of EDI that does not depend on c, so `edi_select_c` computes it once
    and reuses it for every candidate c.

    Parameters
    ----------
    events : structured array (x, y, t, p), t in microseconds, p in {0, 1}. Events outside
        [t_begin, t_end) are ignored, so the whole recording can be passed in.
    t_begin, t_end : exposure window, microseconds
    t_ref : time the sharp image is reconstructed at, t_begin <= t_ref <= t_end
    shape : (H, W)
    n_bins : number of time bins over the exposure (default: see `_default_n_bins`)

    Returns
    -------
    E : (n_bins + 1, H, W) float32. E[k] is the signed count (ON = +1, OFF = -1) of events
        between t_ref and bin edge k: positive direction after t_ref, negated before it.
    n_events : number of events inside the window
    """
    if not t_end > t_begin:
        raise ValueError("t_end must be greater than t_begin")
    if not t_begin <= t_ref <= t_end:
        raise ValueError("t_ref must lie inside the exposure window")
    H, W = shape
    n_bins = _default_n_bins(t_begin, t_end) if n_bins is None else int(n_bins)

    t = events["t"]
    if len(t) and (t[0] > t_begin or t[-1] < t_end) and (t.min() < t_begin or t.max() >= t_end):
        # `events` is a longer stream (it has events outside the window) that starts after
        # or stops before the exposure does: part of the window has no event coverage.
        warnings.warn(f"event stream [{t.min()}, {t.max()}] us covers only part of the exposure "
                      f"window [{t_begin}, {t_end}] us; the uncovered part is treated as no events",
                      stacklevel=3)
    m = (t >= t_begin) & (t < t_end)
    ev = events[m]
    if len(ev) == 0:
        warnings.warn("no events inside the exposure window: EDI returns the input unchanged",
                      stacklevel=3)

    pol = np.where(ev["p"] > 0, 1.0, -1.0)
    pix = ev["y"].astype(np.int64) * W + ev["x"].astype(np.int64)
    k = np.floor((ev["t"] - t_begin) * (n_bins / (t_end - t_begin))).astype(np.int64)
    k = np.clip(k, 0, n_bins - 1)

    hist = np.bincount(k * (H * W) + pix, weights=pol, minlength=n_bins * H * W)
    E = np.zeros((n_bins + 1, H, W), dtype=np.float32)
    np.cumsum(hist.reshape(n_bins, H, W), axis=0, out=E[1:])   # signed count from t_begin to each edge
    before = ev["t"] < t_ref
    E -= np.bincount(pix[before], weights=pol[before], minlength=H * W).reshape(H, W)  # re-reference to t_ref
    return E, len(ev)


def _divisor(E, c):
    """mean over the exposure of exp(c * E(t)), trapezoid rule over the bin edges."""
    return np.trapezoid(np.exp(c * E), axis=0) / (E.shape[0] - 1)


def edi_divisor(events, t_begin, t_end, t_ref, c, shape, n_bins=None):
    """EDI divisor: per pixel, the mean over the exposure of exp(c * E(t)).

    E(t) is the signed event count between t_ref and t (negative direction when
    t < t_ref). The blurry frame equals the sharp frame at t_ref times this map.

    Parameters
    ----------
    events : structured array (x, y, t, p), t in microseconds, p in {0, 1}
    t_begin, t_end : exposure window, microseconds
    t_ref : reconstruction time, microseconds, inside the window
    c : contrast threshold (log-intensity change per event)
    shape : (H, W)
    n_bins : time bins over the exposure; default is about one bin per 0.5 ms (see
        `_default_n_bins` for why)

    Returns
    -------
    (H, W) float array, equal to 1 at pixels with no events
    """
    E, _ = event_integral(events, t_begin, t_end, t_ref, shape, n_bins)
    return _divisor(E, c)


def edi_deblur(blurry, events, t_begin, t_end, c, t_ref=None, n_bins=None):
    """Event-based Double Integral deblurring (Pan et al., CVPR 2019).

    Parameters
    ----------
    blurry : (H, W) float image in [0, 1], linear intensity
    events : structured array (x, y, t, p), t in microseconds, p in {0, 1}
    t_begin, t_end : exposure window of `blurry`, microseconds
    c : contrast threshold
    t_ref : time to reconstruct the sharp image at (default: exposure midpoint)
    n_bins : see `edi_divisor`

    Returns
    -------
    (H, W) float image in [0, 1]: blurry / divisor, clipped. With no events in the window
    the divisor is 1 everywhere and the input comes back unchanged (a warning is raised).
    """
    t_ref = (t_begin + t_end) / 2 if t_ref is None else t_ref
    return np.clip(blurry / edi_divisor(events, t_begin, t_end, t_ref, c, blurry.shape, n_bins), 0, 1)


def _grad(img):
    """Forward differences (gx, gy), zero at the last column/row. Works for H = 1."""
    return (np.diff(img, axis=1, append=img[:, -1:]), np.diff(img, axis=0, append=img[-1:]))


def _ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def event_edge_map(events, t_begin, t_end, t_ref, shape, window=0.2):
    """Edge map from the events closest in time to t_ref.

    Events only fire at moving edges, and the ones within a short window around t_ref mark
    where those edges are *at t_ref*, which is where the sharp image should have its
    gradients. Uses the centred fraction `window` of the exposure (all of it if that slice
    is empty), counts events per pixel regardless of polarity and blurs by 1 px.

    Returns
    -------
    (H, W) float array, >= 0
    """
    half = window * (t_end - t_begin) / 2
    t = events["t"]
    m = (t >= max(t_begin, t_ref - half)) & (t < min(t_end, t_ref + half))
    if not m.any():
        m = (t >= t_begin) & (t < t_end)
    H, W = shape
    cnt = np.bincount(events["y"][m].astype(np.int64) * W + events["x"][m].astype(np.int64),
                      minlength=H * W).reshape(H, W).astype(np.float64)
    return cv2.GaussianBlur(cnt, (0, 0), 1.0)


def _search_1d(f, lo, hi, n_coarse, tol):
    """Minimize f on [lo, hi]: coarse grid, then golden-section between the best point's neighbours.

    Returns the minimizer and every (x, f(x)) evaluated, sorted by x.
    """
    xs = list(np.linspace(lo, hi, n_coarse))
    fs = [f(x) for x in xs]
    i = int(np.argmin(fs))
    a, b = xs[max(i - 1, 0)], xs[min(i + 1, n_coarse - 1)]
    x1, x2 = b - GOLDEN * (b - a), a + GOLDEN * (b - a)
    f1, f2 = f(x1), f(x2)
    xs += [x1, x2]; fs += [f1, f2]
    while b - a > tol:
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - GOLDEN * (b - a); f1 = f(x1); xs.append(x1); fs.append(f1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + GOLDEN * (b - a); f2 = f(x2); xs.append(x2); fs.append(f2)
    order = np.argsort(xs)
    xs, fs = np.asarray(xs)[order], np.asarray(fs)[order]
    return float(xs[np.argmin(fs)]), xs, fs


def edi_select_c(blurry, events, t_begin, t_end, t_ref=None, c_range=(0.05, 0.6), n_coarse=12,
                 lam=1.0, tol=2e-3, n_bins=None, edge_window=0.2):
    """Choose the contrast threshold c without ground truth, by the EDI paper's criterion.

    Minimizes  J(c) = TV(L_c) / TV(B)  -  lam * corr(|grad L_c|, M)
    where L_c is the EDI result, B the blurry input and M the event edge map
    (`event_edge_map`). The correlation term rewards a result whose edges sit where the
    events say the edges are at t_ref; total variation penalizes the noise and ringing that
    a too-large c amplifies. Compared with the paper: TV is divided by the input's TV and
    the edge term is a normalized correlation, so that both are dimensionless and `lam`
    does not depend on image size or brightness.

    Search: `n_coarse` evenly spaced values over `c_range`, then golden-section refinement
    between the neighbours of the best coarse value, down to width `tol`.

    Parameters
    ----------
    blurry, events, t_begin, t_end, t_ref, n_bins : as in `edi_deblur`
    c_range : (c_min, c_max) searched
    lam : weight of the edge-correlation term
    edge_window : fraction of the exposure around t_ref used for the event edge map

    Returns
    -------
    c : selected contrast threshold
    curve : dict of arrays sorted by c, one entry per c evaluated:
        "c", "objective", "tv" (relative TV), "corr" (edge correlation)
    """
    t_ref = (t_begin + t_end) / 2 if t_ref is None else t_ref
    E, n = event_integral(events, t_begin, t_end, t_ref, blurry.shape, n_bins)
    M = event_edge_map(events, t_begin, t_end, t_ref, blurry.shape, edge_window)
    tv0 = sum(np.abs(g).sum() for g in _grad(blurry)) + 1e-12
    terms = {}

    def objective(c):
        gx, gy = _grad(np.clip(blurry / _divisor(E, c), 0, 1))
        tv = (np.abs(gx).sum() + np.abs(gy).sum()) / tv0
        corr = _ncc(np.hypot(gx, gy), M) if n else 0.0
        terms[c] = (tv, corr)
        return tv - lam * corr

    c, cs, obj = _search_1d(objective, *c_range, n_coarse, tol)
    return c, {"c": cs, "objective": obj,
               "tv": np.array([terms[x][0] for x in cs]), "corr": np.array([terms[x][1] for x in cs])}


def edi_oracle_c(blurry, sharp, events, t_begin, t_end, t_ref=None, c_range=(0.0, 0.6), n_coarse=13,
                 tol=2e-3, n_bins=None):
    """ORACLE: the c that maximizes PSNR against the ground-truth sharp image.

    Needs ground truth, so it is an upper bound on what any c-selection rule can reach with
    this model, not a usable method. Same search as `edi_select_c`.

    Returns
    -------
    c : PSNR-maximizing contrast threshold
    curve : dict of arrays sorted by c: "c", "psnr" (dB)
    """
    t_ref = (t_begin + t_end) / 2 if t_ref is None else t_ref
    E, _ = event_integral(events, t_begin, t_end, t_ref, blurry.shape, n_bins)
    mse = lambda c: float(np.mean((np.clip(blurry / _divisor(E, c), 0, 1) - sharp) ** 2))
    c, cs, errs = _search_1d(mse, *c_range, n_coarse, tol)
    return c, {"c": cs, "psnr": -10 * np.log10(errs + 1e-12)}
