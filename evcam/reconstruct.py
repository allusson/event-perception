"""Intensity reconstruction from events only: direct integration and the high-pass filter.

Both return the per-pixel log intensity change l_hat at the query times, relative to an
unknown per-pixel starting level, so they are compared with ground truth after
`fit_scale_offset`.
"""

import numpy as np


def _signed_weights(events, c):
    """c_on for ON events, -c_off for OFF events (c scalar or (c_on, c_off))."""
    c_on, c_off = (c, c) if np.isscalar(c) else c
    return np.where(events["p"] > 0, float(c_on), -float(c_off))


def _sorted(events):
    if len(events) and np.any(np.diff(events["t"]) < 0):
        events = events[np.argsort(events["t"], kind="stable")]
    return events


def direct_integration(events, shape, c, t_query):
    """Per-pixel sum of the events up to each query time, l_hat(t) = c * sum_{t_k <= t} p_k.

    Errors in c, missed events and noise events add up without bound, so l_hat drifts;
    this is the baseline that shows why a filter is needed.

    Parameters
    ----------
    events : structured array (x, y, t, p), t in microseconds, p in {0, 1}
    shape : (H, W)
    c : contrast threshold, scalar or (c_on, c_off)
    t_query : (Q,) query times, microseconds, in increasing order

    Returns
    -------
    (Q, H, W) float32, log intensity change since the start of the stream
    """
    H, W = shape
    events = _sorted(events)
    w = _signed_weights(events, c)
    pix = events["y"].astype(np.int64) * W + events["x"].astype(np.int64)
    idx = np.searchsorted(events["t"], np.asarray(t_query), side="right")
    out = np.empty((len(idx), H, W), np.float32)
    state, i_prev = np.zeros(H * W), 0
    for q, i in enumerate(idx):
        state += np.bincount(pix[i_prev:i], weights=w[i_prev:i], minlength=H * W)
        out[q], i_prev = state.reshape(H, W), i
    return out


def highpass_filter(events, shape, c, cutoff_hz, t_query):
    """High-pass filter of Scheerlinck et al. (ACCV 2018), events only.

    Per pixel,
        d l_hat / dt = sum_k c p_k delta(t - t_k) - alpha * l_hat,   alpha = 2 pi f_c,
    so between events l_hat decays as exp(-alpha dt) and at each event it jumps by c p_k.

    The solution is evaluated exactly at the query times, from one query time T' to the
    next T:
        l_hat(T) = l_hat(T') exp(-alpha (T - T')) + sum_{T' < t_k <= T} c p_k exp(-alpha (T - t_k))
    Each step is one weighted bincount over the events of the interval, so the cost is
    O(events + queries * pixels) with no per-event Python loop. The state starts at 0.

    Parameters
    ----------
    events : structured array (x, y, t, p), t in microseconds, p in {0, 1}
    shape : (H, W)
    c : contrast threshold, scalar or (c_on, c_off)
    cutoff_hz : f_c; 0 gives direct integration
    t_query : (Q,) query times, microseconds, in increasing order

    Returns
    -------
    (Q, H, W) float32, high-passed log intensity
    """
    H, W = shape
    events = _sorted(events)
    alpha = 2 * np.pi * cutoff_hz * 1e-6                  # per microsecond
    w = _signed_weights(events, c)
    pix = events["y"].astype(np.int64) * W + events["x"].astype(np.int64)
    t = events["t"].astype(np.float64)
    t_query = np.asarray(t_query, dtype=np.float64)
    idx = np.searchsorted(t, t_query, side="right")
    out = np.empty((len(idx), H, W), np.float32)
    state, i_prev, T_prev = np.zeros(H * W), 0, None
    for q, (i, T) in enumerate(zip(idx, t_query)):
        if T_prev is not None:
            state *= np.exp(-alpha * (T - T_prev))
        if i > i_prev:
            state += np.bincount(pix[i_prev:i], weights=w[i_prev:i] * np.exp(-alpha * (T - t[i_prev:i])),
                                 minlength=H * W)
        out[q], i_prev, T_prev = state.reshape(H, W), i, T
    return out


def fit_scale_offset(log_img, ref):
    """Map a log-intensity reconstruction to the reference by a least-squares scale and offset.

    Reconstructions from events are log intensity up to an unknown offset and scale, so
    I_hat = a * exp(l_hat) + b is fitted to the reference frame (linear intensity) over all
    pixels, and I_hat is returned clipped to [0, 1] (along with a, b).

    Parameters
    ----------
    log_img : (H, W) reconstruction, log intensity
    ref : (H, W) ground truth, linear intensity in [0, 1]

    Returns
    -------
    fitted : (H, W) float, a * exp(log_img) + b clipped to [0, 1]
    (a, b) : the fitted scale and offset
    """
    x = np.exp(np.asarray(log_img, np.float64)).ravel()
    A = np.stack([x, np.ones_like(x)], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, np.asarray(ref, np.float64).ravel(), rcond=None)
    return np.clip(a * np.exp(log_img) + b, 0, 1), (float(a), float(b))


def reconstruct_e2vid(events, shape, t_query, weights_path="weights/E2VID_lightweight.pth.tar", device=None):
    """Not implemented: E2VID (Rebecq et al., TPAMI 2019) reconstruction.

    The reference implementation (github.com/uzh-rpg/rpg_e2vid) is GPL-3.0 licensed,
    unlike EFNet (Apache-2.0), so it is not vendored into `evcam`. To use it: clone that
    repo outside this package, download the weights

        https://rpg.ifi.uzh.ch/data/E2VID/models/E2VID_lightweight.pth.tar

    into `weights/`, load them with the repo's `utils.loading_utils.load_model`, and feed it
    voxel grids (5 bins) of the events between consecutive query times, keeping the
    recurrent state between calls. The output is an intensity image in [0, 1] per query
    time, compared with ground truth after `fit_scale_offset` on its log.
    """
    raise NotImplementedError("E2VID is not vendored (rpg_e2vid is GPL-3.0); see reconstruct_e2vid's docstring.")
