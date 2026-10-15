"""Frequency estimation from events: event-rate spectrum and per-pixel event intervals."""

import bisect

import cv2
import numpy as np

from evcam.io import EVENT_DTYPE

# Minimum local peak ratio (see `local_ratio`) for a spectrum peak to count as detected.
# Measured with the notebook 04 settings: 5.9e6 (synthetic 437 Hz), 210 / 15488 / 34885 / 3694 / 180
# (LED 10 / 50 / 500 / 1000 / 3000 Hz, ROI) and at most 21.7 for the strongest peak of
# 40x40 patches away from the LED (noise only). 60 is about the geometric mean of 21.7 and 180.
SNR_MIN = 60.0


def _in_roi(events, roi):
    x0, y0, x1, y1 = roi
    return (events["x"] >= x0) & (events["x"] < x1) & (events["y"] >= y0) & (events["y"] < y1)


def busiest_roi(events, shape, size=40):
    """Square ROI (x0, y0, x1, y1) of side `size` centred on the pixel with the most events.

    The per-pixel event count is smoothed with a 5x5 box blur before the maximum is taken,
    so a single hot pixel does not win. Near the border the square is shifted to stay
    inside the frame.

    Parameters
    ----------
    events : structured array (x, y, t, p)
    shape : (H, W)
    size : side of the square, pixels

    Returns
    -------
    (x0, y0, x1, y1) in pixels, x1 and y1 exclusive
    """
    H, W = shape
    counts = np.bincount(events["y"].astype(np.int64) * W + events["x"].astype(np.int64),
                         minlength=H * W).reshape(H, W).astype(np.float32)
    cy, cx = np.unravel_index(cv2.blur(counts, (5, 5), borderType=cv2.BORDER_CONSTANT).argmax(), (H, W))
    x0 = int(np.clip(cx - size // 2, 0, max(W - size, 0)))
    y0 = int(np.clip(cy - size // 2, 0, max(H - size, 0)))
    return x0, y0, min(x0 + size, W), min(y0 + size, H)


def event_rate_spectrum(events, t_start, t_end, bin_us=100, roi=None, f_min=5.0, polarity=None,
                        rel_height=0.5, snr_min=None, return_ratio=False):
    """Power spectrum of the event count over time, and its peak frequency.

    Events in [t_start, t_end) are counted in bins of `bin_us`, the mean is removed, a Hann
    window is applied and the squared magnitude of the FFT is returned. Frequencies up to
    1 / (2 * bin_us) are resolved, in steps of 1 / (t_end - t_start).

    Each local maximum above `f_min` is rated by its `local_ratio`: its power over the median
    power within +-20 % of its frequency. A narrow flicker line has a high local ratio, a
    broad drift bump a low one. Candidates are the peaks with local_ratio >= `snr_min`.

    A periodic train of short event bursts has harmonics almost as strong as its
    fundamental, so the plain argmax can land on a multiple of the true frequency. The peak
    returned here is therefore the lowest-frequency candidate whose power reaches
    `rel_height` times that of the strongest candidate (`rel_height=1` gives the strongest
    candidate). If no peak is a candidate, the peak with the highest local_ratio is returned
    (it is then not detected). The peak is refined to sub-bin accuracy with a parabola
    through the log power of the peak bin and its two neighbours.

    Parameters
    ----------
    events : structured array (x, y, t, p), t in microseconds
    t_start, t_end : window, microseconds
    bin_us : width of the counting bins, microseconds
    roi : optional (x0, y0, x1, y1) in pixels, x1 and y1 exclusive; only events inside count
    f_min : peaks below this frequency (Hz) are ignored
    polarity : None (all events), or 1 / 0 to count only ON / OFF events. A flicker fires
        ON events on the rising and OFF events on the falling edge, so with both
        polarities and a 50 % duty cycle the event rate repeats at twice the frequency.
    rel_height : see above
    snr_min : see above; None uses the module default SNR_MIN
    return_ratio : also return the peak quality `peak_ratio`

    Returns
    -------
    freqs : (K,) frequencies in Hz
    power : (K,) power spectrum (arbitrary units)
    f_peak : peak frequency in Hz (NaN if the window holds no usable peak)
    peak_ratio : only if `return_ratio`; `local_ratio` of the peak, NaN if there is no
        peak. The peak counts as detected if peak_ratio >= SNR_MIN (see `peak_detected`).
    """
    m = (events["t"] >= t_start) & (events["t"] < t_end)
    if roi is not None:
        m &= _in_roi(events, roi)
    if polarity is not None:
        m &= events["p"] == polarity
    n_bins = int((t_end - t_start) // bin_us)
    counts = np.bincount((events["t"][m] - t_start) // bin_us, minlength=n_bins)[:n_bins].astype(float)

    power = np.abs(np.fft.rfft((counts - counts.mean()) * np.hanning(n_bins))) ** 2
    freqs = np.fft.rfftfreq(n_bins, d=bin_us * 1e-6)

    # local maxima above f_min (the last bin has no right neighbour and is left out)
    k = np.arange(1, len(power) - 1)
    k = k[(freqs[k] >= f_min) & (power[k] > power[k - 1]) & (power[k] >= power[k + 1])]
    ratio = local_ratio(freqs, power, k)
    k, ratio = k[np.isfinite(ratio)], ratio[np.isfinite(ratio)]
    if len(k) == 0:
        return (freqs, power, np.nan, np.nan) if return_ratio else (freqs, power, np.nan)
    cand = ratio >= (SNR_MIN if snr_min is None else snr_min)
    if cand.any():
        k, ratio = k[cand], ratio[cand]
        i = np.flatnonzero(power[k] >= rel_height * power[k].max())[0]
    else:
        i = ratio.argmax()
    k, ratio = k[i], ratio[i]
    a, b, c = np.log(power[k - 1:k + 2] + 1e-300)
    shift = 0.5 * (a - c) / (a - 2 * b + c)
    f_peak = freqs[k] + shift * (freqs[1] - freqs[0])
    return (freqs, power, f_peak, ratio) if return_ratio else (freqs, power, f_peak)


def local_ratio(freqs, power, k, rel_width=0.2, exclude=3):
    """Local peak quality P(f_k) / median of P over [(1 - rel_width) f_k, (1 + rel_width) f_k].

    The `exclude` bins on each side of k (and k itself) are left out of the median, so a
    peak a few bins wide does not raise its own reference. NaN where nothing is left.

    Parameters
    ----------
    freqs, power : (K,) spectrum from `event_rate_spectrum`
    k : (n,) int indices of the peaks to rate
    rel_width : half-width of the reference band, relative to f_k
    exclude : bins left out on each side of k

    Returns
    -------
    (n,) float array
    """
    k = np.asarray(k, dtype=np.int64)
    lo = np.searchsorted(freqs, (1 - rel_width) * freqs[k], side="left")
    hi = np.searchsorted(freqs, (1 + rel_width) * freqs[k] + 1e-6 * freqs[1], side="right")
    out = np.full(len(k), np.nan)
    # The bands only move up with k, so one sorted list of the band's powers is kept and
    # updated; the median of the band without the excluded bins is read from it by rank.
    order = np.argsort(k, kind="stable")
    band, a, b = [], 0, 0
    for i in order:
        kk = k[i]
        while b < hi[i]:
            bisect.insort(band, power[b]); b += 1
        while a < lo[i]:
            del band[bisect.bisect_left(band, power[a])]; a += 1
        hole = np.sort(power[max(kk - exclude, a):min(kk + exclude + 1, b)])
        n = len(band) - len(hole)
        if n <= 0:
            continue
        pos = sorted(bisect.bisect_left(band, v) + j - np.searchsorted(hole, v) for j, v in enumerate(hole))

        def nth(r):                     # r-th smallest of the band without the hole
            for p in pos:
                if p <= r:
                    r += 1
            return band[r]

        med = 0.5 * (nth((n - 1) // 2) + nth(n // 2))
        out[i] = power[kk] / med if med > 0 else np.inf
    return out


def peak_detected(peak_ratio, snr_min=SNR_MIN):
    """True if a spectrum peak with this peak_ratio counts as detected (NaN gives False)."""
    return bool(peak_ratio >= snr_min)


def block_frequency_map(events, shape, t0, t1, block=10, bin_us=50, polarity=1, min_events=20, snr_min=SNR_MIN):
    """Peak frequency of the pooled event-rate spectrum of each block x block pixel tile.

    The events of each tile are pooled and passed to `event_rate_spectrum`, so a tile gets
    a value as long as some of its pixels fire on each cycle, even if no single pixel does.

    Parameters
    ----------
    events : structured array (x, y, t, p), t in microseconds
    shape : (H, W)
    t0, t1 : window, microseconds
    block : tile side, pixels; tile (i, j) covers rows i*block ... and columns j*block ...
    bin_us : width of the counting bins, microseconds
    polarity : None (all events), or 1 / 0 for ON / OFF events only
    min_events : tiles with fewer events (of that polarity, in the window) are NaN
    snr_min : tiles whose peak_ratio is below this are NaN

    Returns
    -------
    (ceil(H / block), ceil(W / block)) float array, frequency in Hz, NaN where there are too
    few events or no detected peak
    """
    H, W = shape
    nby, nbx = -(-H // block), -(-W // block)
    m = (events["t"] >= t0) & (events["t"] < t1)
    if polarity is not None:
        m &= events["p"] == polarity
    ev = events[m]
    tile = (ev["y"].astype(np.int64) // block) * nbx + ev["x"].astype(np.int64) // block
    order = np.argsort(tile, kind="stable")
    ev, tile = ev[order], tile[order]
    n = np.bincount(tile, minlength=nby * nbx)
    start = np.cumsum(n) - n

    fmap = np.full(nby * nbx, np.nan)
    for i in np.flatnonzero(n >= min_events):
        _, _, f_peak, ratio = event_rate_spectrum(ev[start[i]:start[i] + n[i]], t0, t1, bin_us=bin_us,
                                                  snr_min=snr_min, return_ratio=True)
        if peak_detected(ratio, snr_min):
            fmap[i] = f_peak
    return fmap.reshape(nby, nbx)


def pixel_frequency_map(events, shape, polarity=1, min_events=10, refractory_us=0):
    """Per-pixel frequency from the median interval between consecutive same-polarity events.

    Parameters
    ----------
    events : structured array (x, y, t, p), t in microseconds
    shape : (H, W)
    polarity : 1 (ON) or 0 (OFF); only these events are used
    min_events : pixels with fewer events of that polarity are NaN
    refractory_us : an event closer than this to the previous event of its pixel is dropped
        first. A strong edge fires a burst of several events; without this the median
        interval is the spacing inside the burst, not the period. 0 keeps every event.

    Returns
    -------
    (H, W) float array, frequency in Hz, NaN where there are too few events
    """
    H, W = shape
    ev = events[events["p"] == polarity]
    pix = ev["y"].astype(np.int64) * W + ev["x"].astype(np.int64)
    order = np.lexsort((ev["t"], pix))
    pix, t = pix[order], ev["t"][order].astype(np.int64)

    if refractory_us > 0 and len(t):
        keep = np.r_[True, (pix[1:] != pix[:-1]) | (np.diff(t) >= refractory_us)]
        pix, t = pix[keep], t[keep]

    n_events = np.bincount(pix, minlength=H * W)
    same = pix[1:] == pix[:-1]
    dt, dpix = np.diff(t)[same], pix[1:][same]

    # median interval per pixel: sort the intervals within each pixel, take the middle one(s)
    order = np.lexsort((dt, dpix))
    dt = dt[order]
    n = np.bincount(dpix, minlength=H * W)
    start = np.cumsum(n) - n
    ok = (n_events >= min_events) & (n > 0)
    lo, hi = start[ok] + (n[ok] - 1) // 2, start[ok] + n[ok] // 2
    freq = np.full(H * W, np.nan)
    freq[ok] = 1e6 / np.maximum(0.5 * (dt[lo] + dt[hi]), 1e-9)
    return freq.reshape(H, W)


def synthetic_flicker(freq_hz, duration_s=1.0, shape=(260, 346), region=(150, 110, 190, 150),
                      duty=0.5, jitter_us=100.0, burst=1, burst_spacing_us=50.0, noise_rate_hz=20000.0,
                      seed=0):
    """Synthetic events from a pixel region flickering at a known frequency.

    Every pixel of the region fires `burst` ON events at each rising edge (once per period)
    and `burst` OFF events at each falling edge, `duty` of a period later. Each event time
    gets Gaussian jitter, and uniformly random noise events are added over the whole frame.

    Parameters
    ----------
    freq_hz : flicker frequency, Hz
    duration_s : length of the recording, seconds
    shape : (H, W) of the sensor (default: DAVIS346)
    region : (x0, y0, x1, y1) flickering pixels, x1 and y1 exclusive
    duty : fraction of the period the region is bright
    jitter_us : standard deviation of the event time jitter, microseconds
    burst : events per edge
    burst_spacing_us : time between consecutive events of a burst, microseconds
    noise_rate_hz : noise events per second over the whole frame
    seed : random seed

    Returns
    -------
    structured array (x, y, t, p) sorted by time, t in microseconds starting at 0
    """
    rng = np.random.default_rng(seed)
    H, W = shape
    x0, y0, x1, y1 = region
    T = duration_s * 1e6
    period = 1e6 / freq_hz

    ys, xs = np.mgrid[y0:y1, x0:x1]
    cycles = np.arange(int(duration_s * freq_hz)) * period
    xs_, ys_, ts_, ps_ = [], [], [], []
    for pol, offset in ((1, 0.0), (0, duty * period)):
        for b in range(burst):
            t = cycles[None, :] + offset + burst_spacing_us * b + rng.normal(0, jitter_us, (xs.size, len(cycles)))
            xs_.append(np.repeat(xs.ravel(), len(cycles)))
            ys_.append(np.repeat(ys.ravel(), len(cycles)))
            ts_.append(t.ravel())
            ps_.append(np.full(t.size, pol))

    n_noise = rng.poisson(noise_rate_hz * duration_s)
    xs_.append(rng.integers(0, W, n_noise))
    ys_.append(rng.integers(0, H, n_noise))
    ts_.append(rng.uniform(0, T, n_noise))
    ps_.append(rng.integers(0, 2, n_noise))

    t = np.concatenate(ts_)
    m = (t >= 0) & (t < T)
    events = np.empty(m.sum(), dtype=EVENT_DTYPE)
    events["x"], events["y"] = np.concatenate(xs_)[m], np.concatenate(ys_)[m]
    events["t"], events["p"] = np.round(t[m]), np.concatenate(ps_)[m]
    return events[np.argsort(events["t"], kind="stable")]
