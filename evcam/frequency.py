"""Frequency estimation from events: event-rate spectrum and per-pixel event intervals."""

import numpy as np

from evcam.io import EVENT_DTYPE


def _in_roi(events, roi):
    x0, y0, x1, y1 = roi
    return (events["x"] >= x0) & (events["x"] < x1) & (events["y"] >= y0) & (events["y"] < y1)


def event_rate_spectrum(events, t_start, t_end, bin_us=100, roi=None, f_min=5.0, polarity=None,
                        rel_height=0.5):
    """Power spectrum of the event count over time, and its peak frequency.

    Events in [t_start, t_end) are counted in bins of `bin_us`, the mean is removed, a Hann
    window is applied and the squared magnitude of the FFT is returned. Frequencies up to
    1 / (2 * bin_us) are resolved, in steps of 1 / (t_end - t_start).

    A periodic train of short event bursts has harmonics almost as strong as its
    fundamental, so the plain argmax can land on a multiple of the true frequency. The peak
    returned here is therefore the lowest-frequency local maximum that reaches `rel_height`
    times the strongest one (`rel_height=1` gives the plain argmax), refined to sub-bin
    accuracy with a parabola through the log power of the peak bin and its two neighbours.

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

    Returns
    -------
    freqs : (K,) frequencies in Hz
    power : (K,) power spectrum (arbitrary units)
    f_peak : peak frequency in Hz (NaN if the window holds no usable peak)
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
    if len(k) == 0:
        return freqs, power, np.nan
    k = k[power[k] >= rel_height * power[k].max()][0]
    a, b, c = np.log(power[k - 1:k + 2] + 1e-300)
    shift = 0.5 * (a - c) / (a - 2 * b + c)
    return freqs, power, freqs[k] + shift * (freqs[1] - freqs[0])


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
                      duty=0.5, jitter_us=100.0, burst=1, noise_rate_hz=20000.0, seed=0):
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
    burst : events per edge, 50 microseconds apart
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
            t = cycles[None, :] + offset + 50.0 * b + rng.normal(0, jitter_us, (xs.size, len(cycles)))
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
