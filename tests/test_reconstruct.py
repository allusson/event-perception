"""Event-only reconstruction on single pixels with known answers."""

import numpy as np

from evcam.io import EVENT_DTYPE
from evcam.reconstruct import direct_integration, fit_scale_offset, highpass_filter

C = 0.2


def events_at(ts, ps, x=0, y=0):
    ev = np.zeros(len(ts), EVENT_DTYPE)
    ev["x"], ev["y"], ev["t"], ev["p"] = x, y, ts, ps
    return ev


def test_highpass_single_on_event_decays_exponentially():
    f_c = 50.0
    alpha = 2 * np.pi * f_c
    t_q = np.array([0, 999, 1000, 3000, 10_000, 31_000])         # event at 1000 us
    out = highpass_filter(events_at([1000], [1]), (1, 1), C, f_c, t_q)[:, 0, 0]
    expected = np.where(t_q >= 1000, C * np.exp(-alpha * (t_q - 1000) / 1e6), 0.0)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-9)


def test_highpass_zero_cutoff_is_direct_integration():
    rng = np.random.default_rng(0)
    ev = events_at(np.sort(rng.integers(0, 10_000, 200)), rng.integers(0, 2, 200), x=rng.integers(0, 3, 200))
    t_q = np.arange(0, 10_001, 500)
    np.testing.assert_allclose(highpass_filter(ev, (1, 3), C, 0.0, t_q), direct_integration(ev, (1, 3), C, t_q),
                               atol=1e-5)


def test_direct_integration_on_off_pairs_return_to_zero():
    ts = [100, 200, 300, 400, 500, 600]
    ps = [1, 0, 1, 1, 0, 0]
    out = direct_integration(events_at(ts, ps), (1, 1), C, [150, 250, 450, 650])[:, 0, 0]
    np.testing.assert_allclose(out, [C, 0, 2 * C, 0], atol=1e-6)


def test_fit_scale_offset_recovers_affine_map():
    rng = np.random.default_rng(1)
    log_img = rng.normal(0, 0.3, (20, 30))
    ref = 0.4 * np.exp(log_img) + 0.05
    fitted, (a, b) = fit_scale_offset(log_img, ref)
    assert abs(a - 0.4) < 1e-9 and abs(b - 0.05) < 1e-9
    np.testing.assert_allclose(fitted, np.clip(ref, 0, 1))
