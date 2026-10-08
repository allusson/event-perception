import numpy as np
import pytest

from evcam.frequency import event_rate_spectrum, pixel_frequency_map, synthetic_flicker

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
