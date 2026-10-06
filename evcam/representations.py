"""Event representations that are not already provided by tonic."""

import numpy as np


def time_surface(events, sensor_size, tau=5000.0):
    """Exponentially decaying time surface, separately for ON and OFF events.

    The reference time is the last event in `events`; each pixel holds
    exp(-(t_ref - t_last) / tau) for the most recent event at that pixel (0 if none).

    Parameters
    ----------
    events : structured array (x, y, t, p), t in microseconds, p in {0, 1}
    sensor_size : (W, H, 2)
    tau : decay constant in microseconds

    Returns
    -------
    ts_on, ts_off : two (H, W) float arrays with values in [0, 1]
    """
    W, H, _ = sensor_size
    t_ref = events["t"][-1]  # reference time: last event in the sample
    decay = np.exp(-(t_ref - events["t"]) / tau)
    surfaces = []
    for pol in (1, 0):
        m = events["p"] == pol
        ts = np.zeros((H, W))
        np.maximum.at(ts, (events["y"][m], events["x"][m]), decay[m])
        surfaces.append(ts)
    return surfaces[0], surfaces[1]


def event_image(events, sensor_size):
    """Accumulate events into one signed count image (ON count - OFF count per pixel).

    Parameters
    ----------
    events : structured array (x, y, t, p), p in {0, 1}
    sensor_size : (W, H, 2)

    Returns
    -------
    (H, W) float array
    """
    ev_img = np.zeros((sensor_size[1], sensor_size[0]))
    np.add.at(ev_img, (events["y"], events["x"]), np.where(events["p"] == 1, 1, -1))
    return ev_img


def scer(events, t_begin, t_end, shape, n_bins=6):
    """Symmetric Cumulative Event Representation (Sun et al., EFNet, ECCV 2022).

    The exposure is cut into `n_bins` equal slices and events are accumulated outward from
    the exposure midpoint t_mid, half of the channels toward the start and half toward the
    end. With edges e_k = t_begin + k * (t_end - t_begin) / n_bins:

        channel k <  n_bins/2 : -(signed count of events in [e_k, t_mid))
        channel k >= n_bins/2 : +(signed count of events in [t_mid, e_{k+1}))

    So channel 0 spans the whole first half, channel n_bins/2 - 1 only the slice just
    before the midpoint, and the second half mirrors that. Each channel is c^-1 times the
    log-intensity change between the mid-exposure sharp image and the image at that
    channel's outer edge, the same quantity EDI integrates; the sign flip on the first
    half makes every channel read "change going away from the midpoint".

    Matches `events_to_accumulate_voxel_torch(..., keep_middle=False)` in the EFNet repo,
    which is what the REBlur voxels were made with. One difference: EFNet takes the window
    from the first and last event it found inside the exposure, this function from the
    exposure times themselves (the two agree to within one event interval).

    Parameters
    ----------
    events : structured array (x, y, t, p), p in {0, 1}; events outside [t_begin, t_end)
        are ignored
    t_begin, t_end : exposure window, same unit as events["t"]
    shape : (H, W)
    n_bins : number of channels, must be even (EFNet uses 6)

    Returns
    -------
    (n_bins, H, W) float32, not normalized. EFNet expects it divided by its max abs value.
    """
    if n_bins % 2:
        raise ValueError("n_bins must be even")
    H, W = shape
    t = events["t"]
    m = (t >= t_begin) & (t < t_end)
    ev = events[m]
    pol = np.where(ev["p"] > 0, 1.0, -1.0)
    pix = ev["y"].astype(np.int64) * W + ev["x"].astype(np.int64)
    k = np.clip(np.floor((ev["t"] - t_begin) * (n_bins / (t_end - t_begin))).astype(np.int64), 0, n_bins - 1)
    slices = np.bincount(k * (H * W) + pix, weights=pol, minlength=n_bins * H * W).reshape(n_bins, H, W)

    half = n_bins // 2
    out = np.empty((n_bins, H, W), dtype=np.float32)
    out[:half] = -np.cumsum(slices[:half][::-1], axis=0)[::-1]   # from slice k up to the midpoint
    out[half:] = np.cumsum(slices[half:], axis=0)                # from the midpoint up to slice k
    return out


def scer_mask(scer_voxels):
    """Event mask EFNet takes next to the SCER: 1 where any channel is non-zero.

    Followed by a 3x3 morphological closing to fill single-pixel holes, as in EFNet's data
    preparation script (`make_voxels_real.py`).

    Parameters
    ----------
    scer_voxels : (C, H, W) SCER from `scer`

    Returns
    -------
    (H, W) float array in {0, 1}
    """
    import cv2
    mask = (scer_voxels != 0).any(axis=0).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(np.float64)
