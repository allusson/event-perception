"""ESIM-style event simulator on single pixels with known answers."""

import numpy as np

from evcam.simulate import simulate_events
from evcam.sources import Clip, synthetic_video

C, EPS = 0.2, 1e-3


def ramp_clip(l0, l1, n_frames=8, dt_us=1000):
    """One pixel whose log intensity log(I + EPS) goes linearly from l0 to l1."""
    t = np.arange(n_frames) * dt_us
    ell = np.linspace(l0, l1, n_frames)
    frames = (np.exp(ell) - EPS).astype(np.float64).reshape(-1, 1, 1)
    return Clip(frames=frames, frame_t=t.astype(np.int64), exposure=None, events=None, shape=(1, 1), meta={})


def test_increasing_ramp_emits_three_on_events_at_crossing_times():
    l0 = np.log(0.1 + EPS)
    clip = ramp_clip(l0, l0 + 3.5 * C)
    ev = simulate_events(clip, c=C, eps=EPS).events
    assert len(ev) == 3 and (ev["p"] == 1).all()
    T = clip.frame_t[-1]
    np.testing.assert_allclose(ev["t"], np.round(np.arange(1, 4) / 3.5 * T), atol=1)
    assert simulate_events(clip, c=C, eps=EPS).meta["c_true"] == C


def test_decreasing_ramp_emits_off_events():
    l0 = np.log(0.8 + EPS)
    ev = simulate_events(ramp_clip(l0, l0 - 3.5 * C), c=C, eps=EPS).events
    assert len(ev) == 3 and (ev["p"] == 0).all()


def test_asymmetric_thresholds():
    l0 = np.log(0.3 + EPS)
    ev = simulate_events(ramp_clip(l0, l0 + 1.0), c=(0.25, 0.1), eps=EPS).events
    assert len(ev) == 4 and (ev["p"] == 1).all()


def test_static_clip_emits_nothing():
    clip = synthetic_video(duration_s=0.01, fps=1000, shape=(40, 50), rev_per_s=0.0, square_velocity=(0, 0))
    assert simulate_events(clip, c=C).n_events == 0


def test_refractory_drops_close_events():
    l0 = np.log(0.1 + EPS)
    clip = ramp_clip(l0, l0 + 3.5 * C, n_frames=2, dt_us=700)    # crossings 200 us apart
    ev = simulate_events(clip, c=C, eps=EPS, refractory_us=300).events
    np.testing.assert_allclose(ev["t"], [200, 600], atol=1)
