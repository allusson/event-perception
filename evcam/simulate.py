"""Event simulation from frames, and blurry / sharp / events triplets with known answers."""

import dataclasses

import numpy as np

from evcam.io import EVENT_DTYPE


def _per_pixel(value, sigma, shape, rng):
    """Threshold map: `value` everywhere, plus N(0, sigma) per pixel if sigma > 0 (floored at 0.01)."""
    if sigma <= 0:
        return np.full(shape, float(value))
    return np.maximum(value + sigma * rng.standard_normal(shape), 0.01)


def simulate_events(clip, c=0.2, c_sigma=0.0, refractory_us=0, noise_rate_hz=0.0, eps=1e-3, seed=0):
    """Simulate an ideal DVS from the frames of a clip (ESIM-style model).

    Per pixel, the log intensity is l = log(I + eps). Each pixel keeps a reference level
    l_ref, initialized from the first frame. Between consecutive frames l is linearly
    interpolated in time; each time |l - l_ref| reaches the threshold, an event is emitted
    with polarity = sign of the change, at the interpolated crossing time, and l_ref moves
    by +-c. Several crossings between two frames are allowed; on a linear segment they are
    evenly spaced in time.

    This is the standard model of ESIM (Rebecq et al., CoRL 2018), without its adaptive
    frame sampling. It is only accurate when the motion between frames is small (high
    frame rate): with large motion the true per-pixel signal is not linear between frames,
    and changes that go up and back down between two frames are missed. v2e (Delbruck et
    al., CVPRW 2021) adds frame interpolation, a finite pixel bandwidth and realistic
    noise; see `simulate_events_v2e`.

    Parameters
    ----------
    clip : Clip with frames (linear intensity) and frame_t (microseconds)
    c : contrast threshold, scalar or (c_on, c_off)
    c_sigma : if > 0, each pixel gets its own thresholds drawn from N(c, c_sigma)
        (fabrication mismatch), floored at 0.01
    refractory_us : an event closer than this to the previous emitted event of its pixel
        is dropped; l_ref still moves, so the change it reported is lost
    noise_rate_hz : uniformly random noise events per second over the whole sensor
        (uniform pixel, time and polarity); they do not touch l_ref
    eps : offset inside the log, so black pixels have a finite log intensity
    seed : random seed (threshold map and noise)

    Returns
    -------
    Clip, a copy of `clip` with `events` replaced by the simulated events (sorted by time,
    EVENT_DTYPE, p in {0, 1}) and meta["c_true"] = c as given (scalar or (c_on, c_off)).
    With c_sigma > 0, meta["c_on_map"] and meta["c_off_map"] hold the per-pixel thresholds.
    """
    if clip.frames is None or len(clip.frames) < 2:
        raise ValueError("simulate_events needs a clip with at least 2 frames")
    rng = np.random.default_rng(seed)
    H, W = clip.shape
    c_on, c_off = (c, c) if np.isscalar(c) else c
    c_on_map = _per_pixel(c_on, c_sigma, (H, W), rng).ravel()
    c_off_map = _per_pixel(c_off, c_sigma, (H, W), rng).ravel()

    t = clip.frame_t.astype(np.float64)
    l_prev = np.log(clip.frames[0].astype(np.float64).ravel() + eps)
    ref = l_prev.copy()
    last_t = np.full(H * W, -np.inf)
    xs, ts, ps = [], [], []
    for k in range(1, len(clip.frames)):
        l_next = np.log(clip.frames[k].astype(np.float64).ravel() + eps)
        for pol, sign, cmap in ((1, 1.0, c_on_map), (0, -1.0, c_off_map)):
            # crossings of l_ref + sign * j * c, j = 1..n, on the segment l_prev -> l_next
            n = np.floor(sign * (l_next - ref) / cmap).astype(np.int64)
            pix = np.flatnonzero(n > 0)
            if len(pix) == 0:
                continue
            n = n[pix]
            j = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) + 1       # 1..n per pixel
            p = np.repeat(pix, n)
            level = ref[p] + sign * j * cmap[p]
            # a level reached already at the frame (rounding of l_ref) gets fraction 0
            d = l_next[p] - l_prev[p]
            frac = np.clip((level - l_prev[p]) / np.where(d != 0, d, 1.0), 0.0, 1.0)
            tc = t[k - 1] + frac * (t[k] - t[k - 1])
            ref[pix] += sign * n * cmap[pix]
            if refractory_us > 0:
                keep = np.zeros(len(p), bool)
                start = np.cumsum(n) - n
                for jj in range(n.max()):                 # crossing jj of every pixel that has one
                    sel = np.flatnonzero(n > jj)
                    idx = start[sel] + jj
                    ok = tc[idx] - last_t[pix[sel]] >= refractory_us
                    keep[idx[ok]] = True
                    last_t[pix[sel][ok]] = tc[idx[ok]]
                p, tc = p[keep], tc[keep]
            xs.append(p); ts.append(tc); ps.append(np.full(len(p), pol))
        l_prev = l_next

    if noise_rate_hz > 0:
        n_noise = rng.poisson(noise_rate_hz * (t[-1] - t[0]) / 1e6)
        xs.append(rng.integers(0, H * W, n_noise)); ts.append(rng.uniform(t[0], t[-1], n_noise))
        ps.append(rng.integers(0, 2, n_noise))

    pix = np.concatenate(xs) if xs else np.zeros(0, np.int64)
    tt = np.concatenate(ts) if ts else np.zeros(0)
    events = np.empty(len(pix), EVENT_DTYPE)
    events["x"], events["y"] = pix % W, pix // W
    events["t"], events["p"] = np.round(tt), np.concatenate(ps) if ps else 0
    events = events[np.argsort(events["t"], kind="stable")]

    meta = {**clip.meta, "c_true": c, "c_sigma": c_sigma, "refractory_us": refractory_us,
            "noise_rate_hz": noise_rate_hz, "events_simulated": True}
    if c_sigma > 0:
        meta["c_on_map"], meta["c_off_map"] = c_on_map.reshape(H, W), c_off_map.reshape(H, W)
    return dataclasses.replace(clip, events=events, meta=meta)


def simulate_events_v2e(clip, **kwargs):
    """Not implemented: simulate events with v2e (github.com/SensorsINI/v2e).

    To plug it in: install v2e, convert the clip's frames to 8-bit (or pass linear
    intensity with its linear/log option), feed them with their timestamps to
    `v2ecore.emulator.EventEmulator.generate_events(frame, t_seconds)` frame by frame
    (optionally after its SuperSloMo interpolation to a finer time step), convert the
    returned (t, x, y, p) rows to EVENT_DTYPE with t in microseconds and p in {0, 1}, and
    return `dataclasses.replace(clip, events=..., meta={..., "c_true": (pos_thres, neg_thres)})`.
    Its thresholds, threshold mismatch (sigma_thres), cutoff_hz (pixel bandwidth), leak and
    shot noise rates and refractory period are the kwargs worth exposing.
    """
    raise NotImplementedError("v2e is not installed; see the docstring of simulate_events_v2e.")


def make_blur_triplet(clip, center_idx, n_avg):
    """Blurry frame, sharp ground truth and the events of the exposure, from a high-fps clip.

    The blurry frame is the time average of the clip over the exposure [t_a, t_b], with
    t_a = frame_t[center_idx - n_avg / 2] and t_b = frame_t[center_idx + n_avg / 2]: the
    trapezoid rule over these n_avg + 1 frames (end frames weighted 1/2), which is the
    exact average of the linearly interpolated intensity, and so spans exactly n_avg frame
    periods centred on the sharp frame. The sharp ground truth is the centre frame. The
    events are the clip's (simulated) events with t_a <= t < t_b.

    Even with the true c, EDI is not exact: changes smaller than c since a pixel's last
    event are never reported (quantization), and EDI also assumes the threshold is the same
    everywhere and that every change produced its events.

    Parameters
    ----------
    clip : Clip with frames, frame_t and events (e.g. from `simulate_events`)
    center_idx : index of the sharp frame
    n_avg : even number of frame periods in the exposure

    Returns
    -------
    Clip with frames = the blurry frame (1, H, W), frame_t = [frame_t[center_idx]],
    exposure = [[t_a, t_b]], events in the exposure, and meta "sharp", "c_true", "n_avg",
    "center_idx"
    """
    if n_avg % 2:
        raise ValueError("n_avg must be even so the exposure is centred on a frame")
    a, b = center_idx - n_avg // 2, center_idx + n_avg // 2
    if a < 0 or b >= clip.n_frames:
        raise ValueError(f"frames {a}..{b} are outside the clip (0..{clip.n_frames - 1})")
    w = np.ones(n_avg + 1); w[[0, -1]] = 0.5
    blurry = np.tensordot(w / w.sum(), clip.frames[a:b + 1].astype(np.float64), axes=1).astype(np.float32)
    t_a, t_b = int(clip.frame_t[a]), int(clip.frame_t[b])
    ev = None
    if clip.events is not None:
        i0, i1 = np.searchsorted(clip.events["t"], [t_a, t_b])
        ev = clip.events[i0:i1]
    meta = {k: v for k, v in clip.meta.items() if k not in ("c_on_map", "c_off_map")}
    meta.update(sharp=clip.frames[center_idx].copy(), c_true=clip.meta.get("c_true"), n_avg=n_avg,
                center_idx=center_idx)
    return dataclasses.replace(clip, frames=blurry[None], frame_t=clip.frame_t[[center_idx]].copy(),
                               exposure=np.array([[t_a, t_b]], np.int64), events=ev, meta=meta)
