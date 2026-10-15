import numpy as np
import pytest

from evcam.frequency import (block_frequency_map, busiest_roi, event_rate_spectrum, local_ratio, peak_detected,
                             pixel_frequency_map, synthetic_flicker)

SHAPE, REGION = (260, 346), (150, 110, 190, 150)


@pytest.mark.parametrize("f_true", [120.0, 437.0])
def test_event_rate_spectrum_recovers_frequency(f_true):
    ev = synthetic_flicker(f_true, shape=SHAPE, region=REGION)
    for kwargs in ({"polarity": 1}, {"polarity": 1, "roi": REGION}, {"polarity": 0}):
        freqs, power, f_peak = event_rate_spectrum(ev, 0, 1_000_000, **kwargs)
        assert len(freqs) == len(power)
        assert abs(f_peak - f_true) / f_true < 0.02


@pytest.mark.parametrize("f_true", [120.0, 437.0])
def test_pixel_frequency_map_recovers_frequency(f_true):
    ev = synthetic_flicker(f_true, shape=SHAPE, region=REGION)
    fmap = pixel_frequency_map(ev, SHAPE)
    x0, y0, x1, y1 = REGION
    inside = fmap[y0:y1, x0:x1]
    assert fmap.shape == SHAPE and not np.isnan(inside).any()
    assert np.abs(inside - f_true).max() / f_true < 0.02      # every flickering pixel
    outside = fmap.copy()
    outside[y0:y1, x0:x1] = np.nan
    assert np.isnan(outside).all()                            # noise pixels have too few events


def test_both_polarities_at_half_duty_give_twice_the_frequency():
    """ON at the rising and OFF at the falling edge: the total event rate repeats at 2f."""
    ev = synthetic_flicker(120.0, shape=SHAPE, region=REGION, duty=0.5)
    assert abs(event_rate_spectrum(ev, 0, 1_000_000)[2] - 240.0) < 0.02 * 240.0


def test_refractory_collapses_bursts():
    ev = synthetic_flicker(120.0, shape=SHAPE, region=REGION, burst=3, jitter_us=5.0)
    x0, y0, x1, y1 = REGION
    assert np.nanmedian(pixel_frequency_map(ev, SHAPE)[y0:y1, x0:x1]) > 1000      # spacing inside the burst
    fmap = pixel_frequency_map(ev, SHAPE, refractory_us=1000)
    assert abs(np.nanmedian(fmap[y0:y1, x0:x1]) - 120.0) < 0.02 * 120.0


def test_led_bursts_at_1khz():
    """3 ON events within 50 us at each rising edge: spectrum, auto ROI and burst refractory."""
    f_true, region = 1000.0, (160, 120, 180, 140)
    ev = synthetic_flicker(f_true, shape=SHAPE, region=region, burst=3, burst_spacing_us=20.0, jitter_us=5.0)

    x0, y0, x1, y1 = busiest_roi(ev, SHAPE, size=40)
    assert (x1 - x0, y1 - y0) == (40, 40)
    assert x0 <= region[0] and y0 <= region[1] and x1 >= region[2] and y1 >= region[3]

    f_peak = event_rate_spectrum(ev, 0, 1_000_000, bin_us=50, roi=(x0, y0, x1, y1), polarity=1)[2]
    assert abs(f_peak - f_true) / f_true < 0.02

    fmap = pixel_frequency_map(ev, SHAPE, polarity=1, refractory_us=0.25e6 / f_peak)
    assert abs(np.nanmedian(fmap[y0:y1, x0:x1]) - f_true) / f_true < 0.02


def test_peak_ratio_separates_flicker_from_noise():
    ev = synthetic_flicker(437.0, shape=SHAPE, region=REGION)
    ratio = event_rate_spectrum(ev, 0, 1_000_000, polarity=1, return_ratio=True)[3]
    assert peak_detected(ratio)
    noise = synthetic_flicker(437.0, shape=SHAPE, region=(0, 0, 0, 0))     # noise events only
    ratio = event_rate_spectrum(noise, 0, 1_000_000, polarity=1, return_ratio=True)[3]
    assert not peak_detected(ratio)


def test_block_map_recovers_frequency_when_pixels_skip_cycles():
    """Each pixel fires at the rising edge on a random 20 % of cycles: the per-pixel median
    interval is a multiple of the period, the pooled spectrum of each tile is not."""
    f_true, block = 437.0, 10
    ev = synthetic_flicker(f_true, shape=SHAPE, region=REGION)
    x0, y0, x1, y1 = REGION
    in_region = (ev["x"] >= x0) & (ev["x"] < x1) & (ev["y"] >= y0) & (ev["y"] < y1)
    keep = ~in_region | (ev["p"] == 0) | (np.random.default_rng(1).random(len(ev)) < 0.2)
    ev = ev[keep]

    bmap = block_frequency_map(ev, SHAPE, 0, 1_000_000, block=block)
    assert bmap.shape == (26, 35)
    inside = bmap[y0 // block:y1 // block, x0 // block:x1 // block]
    assert not np.isnan(inside).any()
    assert np.abs(inside - f_true).max() / f_true < 0.02

    fmap = pixel_frequency_map(ev, SHAPE, polarity=1)
    assert abs(np.nanmedian(fmap[y0:y1, x0:x1]) - f_true) / f_true > 0.02


def test_local_ratio_matches_direct_median():
    rng = np.random.default_rng(0)
    freqs = np.arange(2000) * 0.5
    power = rng.exponential(1.0, 2000)
    power[[300, 301, 900]] = power[[300, 301, 900]][0]                 # repeated values
    k = np.arange(10, 1999, 7)
    expected = []
    for kk in k:
        band = np.flatnonzero((freqs >= 0.8 * freqs[kk]) & (freqs <= 1.2 * freqs[kk] + 1e-9))
        ref = power[band[np.abs(band - kk) > 3]]
        expected.append(power[kk] / np.median(ref) if len(ref) else np.nan)
    np.testing.assert_allclose(local_ratio(freqs, power, k), expected)
