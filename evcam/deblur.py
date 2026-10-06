"""Event-based deblurring, three ways: EDI (event physics, no motion model, no learning),
CMax + Richardson-Lucy deconvolution (motion model, no learning) and EFNet (learned).

All functions take the blurry frame as a float (H, W) image in [0, 1] together with the
frame's exposure window [t_begin, t_end) in microseconds, on the same clock as the events.
`evcam.io.load_aedat4` returns both for DAVIS recordings (`exposure_begin_t`,
`exposure_end_t`).
"""

import warnings

import cv2
import numpy as np

from evcam.cmax import patchwise_flow

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


class _SparseIntegral:
    """`_divisor` for many values of c: only pixels that saw an event are recomputed.

    Events touch a small part of the frame, and everywhere else the divisor is exactly 1.
    The c searches evaluate the divisor ~30 times per frame, so this is most of their cost.
    """

    def __init__(self, E):
        self.shape = E.shape[1:]
        flat = E.reshape(E.shape[0], -1)
        self.active = np.flatnonzero((flat != 0).any(axis=0))
        self.E = np.ascontiguousarray(flat[:, self.active])

    def divisor(self, c):
        out = np.ones(self.shape[0] * self.shape[1])
        out[self.active] = np.trapezoid(np.exp(c * self.E), axis=0) / (self.E.shape[0] - 1)
        return out.reshape(self.shape)


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
                 lam=0.25, tol=2e-3, n_bins=None, edge_window=0.2):
    """Choose the contrast threshold c without ground truth, by the EDI paper's criterion.

    Minimizes  J(c) = TV(L_c) / TV(B)  -  lam * corr(|grad L_c|, M)
    where L_c is the EDI result, B the blurry input and M the event edge map
    (`event_edge_map`). The correlation term rewards a result whose edges sit where the
    events say the edges are at t_ref; total variation penalizes the noise and ringing that
    a too-large c amplifies. Compared with the paper: TV is divided by the input's TV and
    the edge term is a normalized correlation, so that both are dimensionless and `lam`
    does not depend on image size or brightness.

    The default `lam` = 0.25 was tuned on the REBlur *train* split (the value that gives
    the best mean PSNR there), never on test data. Be aware of how weak this criterion is
    on REBlur: the edge correlation keeps rising with c well past the best c, because any
    change EDI makes sits on event pixels, so the result flips between "stay near the
    lower bound" and "run to a large c" depending on `lam`, and the selected c is
    uncorrelated with the PSNR-optimal one. Notebook 03 quantifies the gap.

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
    E = _SparseIntegral(E)
    M = event_edge_map(events, t_begin, t_end, t_ref, blurry.shape, edge_window)
    tv0 = sum(np.abs(g).sum() for g in _grad(blurry)) + 1e-12
    terms = {}

    def objective(c):
        gx, gy = _grad(np.clip(blurry / E.divisor(c), 0, 1))
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
    E = _SparseIntegral(E)
    mse = lambda c: float(np.mean((np.clip(blurry / E.divisor(c), 0, 1) - sharp) ** 2))
    c, cs, errs = _search_1d(mse, *c_range, n_coarse, tol)
    return c, {"c": cs, "psnr": -10 * np.log10(errs + 1e-12)}


# ---------------------------------------------------------------------------
# CMax flow + Richardson-Lucy deconvolution
# ---------------------------------------------------------------------------

def linear_psf(d):
    """Normalized linear motion PSF for a blur vector d = (dx, dy) in pixels.

    A line segment of length |d| centred on the kernel centre (so the deconvolved image is
    the one at mid-exposure), drawn with bilinear weights at sub-pixel steps.

    Returns
    -------
    (k, k) float array summing to 1, k odd
    """
    length = float(np.hypot(*d))
    r = int(np.ceil(length / 2)) + 1
    k = 2 * r + 1
    psf = np.zeros((k, k))
    n = max(int(np.ceil(length * 4)), 1)
    s = (np.arange(n) + 0.5) / n - 0.5
    xs, ys = r + s * d[0], r + s * d[1]
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    ax, ay = xs - x0, ys - y0
    for ox, oy, w in ((0, 0, (1 - ax) * (1 - ay)), (1, 0, ax * (1 - ay)), (0, 1, (1 - ax) * ay), (1, 1, ax * ay)):
        np.add.at(psf, (y0 + oy, x0 + ox), w)
    return psf / psf.sum()


def cmax_deblur(blurry, events, t_begin, t_end, patch_size=64, min_events=100, num_iter=10,
                min_blur_px=1.0, event_gate=True, return_info=False, **cmax_kwargs):
    """Deblur by estimating motion with contrast maximization, then deconvolving.

    1. Flow. `evcam.cmax.patchwise_flow` is run on the events inside the exposure, with
       time normalized over the exposure, so each patch's flow comes out in pixels per
       exposure, which is the blur vector itself (speed in px/s times exposure time).
    2. PSF. Each patch gets a linear motion PSF of that length and direction.
    3. Deconvolution. Each tile is deconvolved with Richardson-Lucy (skimage) and the
       tiles are blended back together.
    4. Event gate. The result is kept only near pixels that fired events; elsewhere the
       input is returned untouched.

    Choices:
    - patch_size = 64. The flow is assumed constant inside a patch, so smaller is better
      for the model, but CMax needs enough events to have a peaked objective; at REBlur /
      DAVIS resolution (260 x 320 to 346) a 64 px patch typically holds a few hundred to a
      few thousand events per exposure. Same size as notebook 02.
    - 50% overlap. `patchwise_flow` uses a non-overlapping grid, so it is run 4 times on
      grids shifted by half a patch in x, y and both. Each tile is weighted by a Hann
      window; Hann windows at 50% overlap sum to exactly 1, so the blend needs no
      normalization and every pixel is a smooth mix of the 4 tiles covering it. This is
      what hides the seams between tiles with different PSFs.
    - num_iter = 10. Richardson-Lucy sharpens progressively and amplifies noise and
      ringing as it goes; with a PSF that is only approximately right, more iterations
      mostly add ringing. Picked on a subset of the REBlur train split, with the event
      gate on: mean PSNR 34.65 / 34.82 / 34.88 / 34.57 dB at 3 / 5 / 10 / 20 iterations
      (input: 34.37 dB).
    - event_gate. A pixel with no events during the exposure did not change, so it is not
      blurred and deconvolving it can only add ringing. This matters a lot when a patch
      holds a moving object on a static background, which is most of REBlur: the patch's
      one PSF is the object's, and applying it to the background is wrong. The gate is the
      "any event here" mask, closed with a 3x3 element and feathered with a 2 px Gaussian.
      Without it the result is worse than the input on that train subset (33.03 dB at 10
      iterations).
    - Tiles with `min_events` events or fewer, or with a blur shorter than `min_blur_px`,
      are left as they are: no events means nothing moved there.
    - Each tile is deconvolved with a margin of one blur length (taken from the image,
      reflected at the image border) so the tile itself is free of boundary ringing.

    Limits. This is the weakest physical model of the three methods here:
    - One constant flow per patch: wrong wherever a patch contains two motions (a moving
      object against a static background, depth edges) or the flow varies (rotation, zoom).
    - Linear, constant-speed motion over the exposure: no curved paths, no acceleration.
    - No occlusion handling: deconvolution rings at motion boundaries, where the image is
      not a convolution of anything.
    - CMax itself can pick a wrong flow in patches with few events or a single edge
      orientation (aperture problem), and a wrong PSF makes the result worse than the input.
    - Richardson-Lucy assumes Poisson noise and a known PSF, and only uses the events
      indirectly, through the flow (and the gate).
    - The event gate hides the damage outside moving edges but does not fix the model:
      inside the gate a static background pixel next to a moving edge still gets the
      object's PSF.

    Parameters
    ----------
    blurry : (H, W) float image in [0, 1]
    events : structured array (x, y, t, p), t in microseconds
    t_begin, t_end : exposure window, microseconds
    patch_size : side of the square patches (even), pixels
    min_events : patches with this many events or fewer are not deblurred
    num_iter : Richardson-Lucy iterations
    min_blur_px : blur vectors shorter than this are treated as no motion
    event_gate : keep the deconvolved result only near event pixels (see above)
    return_info : also return the estimated motion (see below)
    **cmax_kwargs : passed to `cmax_patch` (pad, rng, coarse). The defaults here search
        +-32 px per exposure.

    Returns
    -------
    (H, W) float image in [0, 1]. With `return_info=True` also a dict with
        "tiles" : list of (x0, y0, dx, dy), one per patch with a flow estimate: top-left
                  corner of the patch and its blur vector in pixels per exposure
        "flow"  : (H, W, 2) Hann-blended blur vector field in pixels per exposure, NaN
                  where no patch had enough events
    """
    from skimage.restoration import richardson_lucy

    H, W = blurry.shape
    ps, half = patch_size, patch_size // 2
    cmax_kwargs = {"pad": 32, "rng": 32, "coarse": 4, **cmax_kwargs}

    m = (events["t"] >= t_begin) & (events["t"] < t_end)
    x, y = events["x"][m].astype(np.float64), events["y"][m].astype(np.float64)
    tn = (events["t"][m] - t_begin) / (t_end - t_begin)

    n = np.arange(ps)
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * (n + 0.5) / ps)
    window = np.outer(hann, hann)

    out = np.zeros((H, W))
    flow_sum, flow_w = np.zeros((H, W, 2)), np.zeros((H, W))
    tiles = []
    for sy in (0, half):
        for sx in (0, half):
            # shift the coordinates so the fixed grid of patchwise_flow lands half a patch over;
            # the canvas is rounded up so that patches hanging over the image edge are included
            shape = (int(np.ceil((H + sy) / ps)) * ps, int(np.ceil((W + sx) / ps)) * ps)
            est = patchwise_flow(x + sx, y + sy, tn, shape, patch_size=ps, min_events=min_events, **cmax_kwargs)
            for py in range(0, shape[0], ps):
                for px in range(0, shape[1], ps):
                    d = est[py, px]
                    # tile in image coordinates, clipped to the image
                    y0, x0 = py - sy, px - sx
                    ya, yb, xa, xb = max(y0, 0), min(y0 + ps, H), max(x0, 0), min(x0 + ps, W)
                    if ya >= yb or xa >= xb:
                        continue
                    w = window[ya - y0:yb - y0, xa - x0:xb - x0]
                    if np.isnan(d[0]) or np.hypot(*d) < min_blur_px:
                        out[ya:yb, xa:xb] += w * blurry[ya:yb, xa:xb]
                        if not np.isnan(d[0]):
                            flow_w[ya:yb, xa:xb] += w
                        continue
                    tiles.append((x0, y0, d[0], d[1]))
                    flow_sum[ya:yb, xa:xb] += w[..., None] * d
                    flow_w[ya:yb, xa:xb] += w

                    mg = int(np.ceil(np.hypot(*d))) + 4          # context margin around the tile
                    ia, ib, ja, jb = max(ya - mg, 0), min(yb + mg, H), max(xa - mg, 0), min(xb + mg, W)
                    region = np.pad(blurry[ia:ib, ja:jb],
                                    ((mg - (ya - ia), mg - (ib - yb)), (mg - (xa - ja), mg - (jb - xb))), mode="reflect")
                    dec = richardson_lucy(np.maximum(region, 1e-4), linear_psf(d), num_iter=num_iter, clip=True)
                    out[ya:yb, xa:xb] += w * dec[mg:mg + yb - ya, mg:mg + xb - xa]

    if event_gate:
        fired = np.bincount((y * W + x).astype(np.int64), minlength=H * W).reshape(H, W) > 0
        fired = cv2.morphologyEx(fired.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        gate = np.clip(cv2.GaussianBlur(fired.astype(np.float64), (0, 0), 2.0), 0, 1)
        out = gate * out + (1 - gate) * blurry
    out = np.clip(out, 0, 1)
    if not return_info:
        return out
    flow = np.where(flow_w[..., None] > 1e-6, flow_sum / np.maximum(flow_w, 1e-6)[..., None], np.nan)
    return out, {"tiles": tiles, "flow": flow}


# ---------------------------------------------------------------------------
# EFNet (Sun et al., ECCV 2022): pretrained inference
# ---------------------------------------------------------------------------

def _torch():
    """Import torch lazily, so EDI and CMax work without the optional [deblur] extra.

    PYTORCH_ENABLE_MPS_FALLBACK must be set before torch is first imported: it lets
    operations that Apple's MPS backend lacks fall back to the CPU instead of raising.
    """
    import os
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch
    return torch


def pick_device():
    """Best available torch device, in the order mps (Apple GPU), cuda, cpu."""
    torch = _torch()
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def efnet_deblur(model, blurry, scer, mask=None, device=None):
    """Deblur one frame with a pretrained EFNet, with the preprocessing of the EFNet repo.

    Preprocessing (same as EFNet's `H5ImageDataset`): image in [0, 1]; SCER divided by its
    own max abs value so it lies in [-1, 1]; mask in {0, 1}. H and W are reflection-padded
    up to multiples of 4 (the network downsamples twice) and the output is cropped back.
    EFNet has two stages and returns both results; the second one is the final image.

    Parameters
    ----------
    model : EFNet from `evcam.efnet_arch.load_efnet`
    blurry : (H, W) float image in [0, 1]. Replicated to 3 channels if the model takes 3
        (the released models do), and the 3 output channels are averaged back to one.
    scer : (6, H, W) SCER from `evcam.representations.scer` or stored in REBlur, raw counts
    mask : (H, W) event mask in {0, 1} (1 where any event fired), or None to run the
        model without its mask-gated connection
    device : torch device (default: `pick_device()`)

    Returns
    -------
    (H, W) float image in [0, 1]
    """
    torch = _torch()
    F = torch.nn.functional
    device = pick_device() if device is None else device
    in_chn = model.conv_01.in_channels
    H, W = blurry.shape

    img = torch.from_numpy(np.ascontiguousarray(blurry, dtype=np.float32))[None, None].repeat(1, in_chn, 1, 1)
    vmax = float(np.abs(scer).max())
    vox = torch.from_numpy(np.ascontiguousarray(scer, dtype=np.float32) / (vmax if vmax > 0 else 1.0))[None]
    inputs = [img, vox]
    if mask is not None:
        inputs.append(torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32))[None, None])

    ph, pw = (-H) % 4, (-W) % 4
    if ph or pw:
        inputs = [F.pad(x, (0, pw, 0, ph), mode="reflect") for x in inputs]
    inputs = [x.to(device) for x in inputs]

    with torch.no_grad():
        out = model(inputs[0], inputs[1], mask=inputs[2] if mask is not None else None)[1]
    out = out[0, :, :H, :W].mean(dim=0)
    return np.clip(out.cpu().numpy().astype(np.float64), 0, 1)
