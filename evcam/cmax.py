"""Contrast maximization (Gallego, Rebecq, Scaramuzza, CVPR 2018) for optical flow.

Throughout this module `tn` is event time normalized to [0, 1) over the window, so a
flow `d` is a displacement in pixels per window.
"""

import cv2
import numpy as np


def iwe(x, y, tn, d, shape, origin=(0, 0), sigma=1.0):
    """Image of Warped Events: warp events back to the window start and accumulate them.

    Each event moves to x' = x - d * tn and splits its weight over the 4 nearest pixels
    (bilinear voting). Polarity is ignored.

    Parameters
    ----------
    x, y : (N,) float pixel coordinates
    tn : (N,) float, time normalized to [0, 1) over the window
    d : flow (dx, dy) in pixels per window; each component is a scalar (one flow for all
        events) or an (N,) array (one flow per event)
    shape : (h, w) of the output canvas
    origin : (x, y) of the canvas' top-left corner in image coordinates, so the IWE can
        be computed on a local canvas around a patch instead of the full frame
    sigma : Gaussian blur in pixels applied to the result (0 = no blur)

    Returns
    -------
    (h, w) float array
    """
    h, w = shape
    xw = x - d[0] * tn - origin[0]
    yw = y - d[1] * tn - origin[1]
    x0, y0 = np.floor(xw).astype(int), np.floor(yw).astype(int)
    ax, ay = xw - x0, yw - y0
    img = np.zeros(h * w)
    for dx, dy, wt in ((0, 0, (1 - ax) * (1 - ay)), (1, 0, ax * (1 - ay)),
                       (0, 1, (1 - ax) * ay), (1, 1, ax * ay)):
        xi, yi = x0 + dx, y0 + dy
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        img += np.bincount(yi[ok] * w + xi[ok], weights=wt[ok], minlength=h * w)
    img = img.reshape(h, w)
    return cv2.GaussianBlur(img, (0, 0), sigma) if sigma > 0 else img


def contrast(img):
    """Contrast objective: variance of an (H, W) image. Returns a float."""
    return img.var()


def cmax_patch(x, y, tn, box, pad=40, rng=40, coarse=4):
    """Constant flow for one patch by coarse-to-fine grid search on IWE variance.

    Searches step `coarse` px over [-rng, rng] in both axes, then step 1 px within
    +-`coarse` of the best coarse point.

    Parameters
    ----------
    x, y : (N,) float pixel coordinates (all events; the patch is selected inside)
    tn : (N,) float, time normalized to [0, 1) over the window
    box : (x0, y0, x1, y1) patch bounds in pixels, x1 and y1 exclusive
    pad : canvas padding in pixels, so events warped past the patch edge aren't dropped
    rng : half-width of the search range, pixels per window
    coarse : coarse search step, pixels

    Returns
    -------
    d : (2,) float array, best flow (dx, dy) in pixels per window
    n : number of events in the patch
    """
    x0, y0, x1, y1 = box
    m = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    xs, ys, ts_ = x[m], y[m], tn[m]
    org, shp = (x0 - pad, y0 - pad), (y1 - y0 + 2 * pad, x1 - x0 + 2 * pad)
    f = lambda d: contrast(iwe(xs, ys, ts_, d, shp, org))
    cands = [(dx, dy) for dx in np.arange(-rng, rng + 1, coarse) for dy in np.arange(-rng, rng + 1, coarse)]
    best = max(cands, key=f)
    fine = np.arange(-coarse, coarse + 1, 1.0)
    best = max(((best[0] + a, best[1] + b) for a in fine for b in fine), key=f)
    return np.array(best, dtype=float), m.sum()


def patchwise_flow(x, y, tn, shape, patch_size=64, min_events=500, **cmax_kwargs):
    """Run `cmax_patch` independently on a grid of non-overlapping square patches.

    Parameters
    ----------
    x, y : (N,) float pixel coordinates
    tn : (N,) float, time normalized to [0, 1) over the window
    shape : (H, W) of the full frame
    patch_size : patch side length in pixels
    min_events : patches with this many events or fewer are skipped
    **cmax_kwargs : passed on to `cmax_patch` (pad, rng, coarse)

    Returns
    -------
    est : (H, W, 2) float array, flow (dx, dy) in pixels per window, constant within
        each patch; NaN in skipped patches and in any border not covered by a full patch
    """
    H, W = shape
    est = np.full((H, W, 2), np.nan)
    for py in range(0, H - patch_size + 1, patch_size):
        for px in range(0, W - patch_size + 1, patch_size):
            d, n = cmax_patch(x, y, tn, (px, py, px + patch_size, py + patch_size), **cmax_kwargs)
            if n > min_events:
                est[py:py + patch_size, px:px + patch_size] = d
    return est
