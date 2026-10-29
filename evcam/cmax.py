"""Contrast maximization (Gallego, Rebecq, Scaramuzza, CVPR 2018) for optical flow.

Throughout this module `tn` is event time normalized to [0, 1) over the window, so a
flow `d` is a displacement in pixels per window.
"""

import cv2
import numpy as np
from scipy.optimize import minimize


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
    return iwe_from_coords(x - d[0] * tn, y - d[1] * tn, shape, origin, sigma)


def iwe_from_coords(xw, yw, shape, origin=(0, 0), sigma=1.0):
    """Image of Warped Events from already-warped coordinates, with the bilinear voting of `iwe`.

    Parameters
    ----------
    xw, yw : (N,) float warped pixel coordinates
    shape : (h, w) of the output canvas
    origin : (x, y) of the canvas' top-left corner in image coordinates
    sigma : Gaussian blur in pixels applied to the result (0 = no blur)

    Returns
    -------
    (h, w) float array
    """
    h, w = shape
    xw = xw - origin[0]
    yw = yw - origin[1]
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


def rotation_warp(x, y, t, omega, M, t_ref):
    """Warp events to t_ref under a constant camera rotation (points at infinity, e.g. stars).

    The direction d = M (x, y, 1) of each event is rotated by exp(-[omega]_x (t - t_ref))
    (Rodrigues formula, exact) and projected back through M^-1, the convention of
    `evcam.io.rotation_flow`.

    Parameters
    ----------
    x, y : (N,) pixel coordinates
    t : (N,) event times, seconds
    omega : (3,) angular velocity, rad/s, in the frame of the directions M (x, y, 1)
    M : (3, 3) camera matrix, pixel (x, y, 1) -> direction
    t_ref : reference time, seconds

    Returns
    -------
    xw, yw : (N,) float warped pixel coordinates
    """
    omega = np.asarray(omega, dtype=float)
    d = np.stack([x, y, np.ones_like(x, dtype=float)], axis=-1) @ M.T
    speed = np.linalg.norm(omega)
    if speed > 0:
        n = omega / speed
        theta = (-speed * (np.asarray(t, dtype=float) - t_ref))[:, None]
        d = (d * np.cos(theta) + np.cross(n, d) * np.sin(theta)
             + n * (d @ n)[:, None] * (1 - np.cos(theta)))
    q = d @ np.linalg.inv(M).T
    return q[:, 0] / q[:, 2], q[:, 1] / q[:, 2]


def cmax_rotation(x, y, t, M, t_ref, canvas, origin, omega0=(0, 0, 0), step=0.05, simplex=0.01):
    """Constant angular velocity maximizing the IWE variance under `rotation_warp`.

    Nelder-Mead is started from omega0 and from omega0 +- `step` on each axis (7 starts);
    the best result is returned.

    Parameters
    ----------
    x, y : (N,) pixel coordinates
    t : (N,) event times, seconds
    M : (3, 3) camera matrix, pixel (x, y, 1) -> direction
    t_ref : reference time, seconds
    canvas : (h, w) of the IWE canvas
    origin : (x, y) of the canvas' top-left corner in image coordinates
    omega0 : (3,) first start, rad/s
    step : offset of the other starts, rad/s
    simplex : size of the initial Nelder-Mead simplex around each start, rad/s

    Returns
    -------
    omega : (3,) float array, rad/s
    contrast : IWE variance at omega
    """
    f = lambda w: -contrast(iwe_from_coords(*rotation_warp(x, y, t, w, M, t_ref), canvas, origin))
    omega0 = np.asarray(omega0, dtype=float)
    starts = [omega0] + [omega0 + s * e for e in np.eye(3) for s in (step, -step)]
    best = None
    for w0 in starts:
        res = minimize(f, w0, method="Nelder-Mead",
                       options={"initial_simplex": np.vstack([w0, w0 + simplex * np.eye(3)]),
                                "xatol": 1e-5, "fatol": 1e-4})
        if best is None or res.fun < best.fun:
            best = res
    return best.x, -best.fun


def rotation_warp_2d(x, y, t, omega, center, t_ref):
    """Warp events to t_ref under a constant in-plane rotation about `center`.

    x' = c + R(-omega (t - t_ref)) (x - c), with R(a) = [[cos a, -sin a], [sin a, cos a]] in
    image coordinates (y down), so omega > 0 is clockwise on screen.

    Parameters
    ----------
    x, y : (N,) pixel coordinates
    t : (N,) event times, seconds
    omega : angular velocity, rad/s
    center : (cx, cy) rotation center, pixels
    t_ref : reference time, seconds

    Returns
    -------
    xw, yw : (N,) float warped pixel coordinates
    """
    a = -omega * (np.asarray(t, dtype=float) - t_ref)
    dx, dy = np.asarray(x, dtype=float) - center[0], np.asarray(y, dtype=float) - center[1]
    ca, sa = np.cos(a), np.sin(a)
    return center[0] + ca * dx - sa * dy, center[1] + sa * dx + ca * dy


def cmax_rotation_2d(x, y, t, center, t_ref, omegas, sigma=1.0, refine=True):
    """Constant in-plane angular velocity maximizing the IWE variance under `rotation_warp_2d`.

    The contrast is evaluated on the grid `omegas`; with `refine`, the best grid point is
    refined by bounded scalar minimization (Brent) between its two grid neighbours. The
    IWE canvas is the square around `center` that holds every event at any rotation.

    Parameters
    ----------
    x, y : (N,) pixel coordinates
    t : (N,) event times, seconds
    center : (cx, cy) rotation center, pixels
    t_ref : reference time, seconds
    omegas : (K,) increasing grid of angular velocities, rad/s (both signs to search both
        rotation directions)
    sigma : Gaussian blur of the IWE, pixels
    refine : refine the best grid point

    Returns
    -------
    omega : best angular velocity, rad/s
    contrasts : (K,) IWE variance at each grid point
    """
    from scipy.optimize import minimize_scalar

    omegas = np.asarray(omegas, dtype=float)
    r = int(np.ceil(np.hypot(np.asarray(x) - center[0], np.asarray(y) - center[1]).max())) + 2
    canvas, origin = (2 * r + 1, 2 * r + 1), (int(np.floor(center[0])) - r, int(np.floor(center[1])) - r)
    f = lambda w: contrast(iwe_from_coords(*rotation_warp_2d(x, y, t, w, center, t_ref), canvas, origin, sigma))
    contrasts = np.array([f(w) for w in omegas])
    k = int(contrasts.argmax())
    omega = omegas[k]
    if refine and len(omegas) > 1:
        lo, hi = omegas[max(k - 1, 0)], omegas[min(k + 1, len(omegas) - 1)]
        res = minimize_scalar(lambda w: -f(w), bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-3 * (hi - lo)})
        if -res.fun > contrasts[k]:
            omega = float(res.x)
    return float(omega), contrasts
