"""Time and space alignment with known answers."""

import numpy as np
import pytest

from evcam.align import check_sync, map_by_keypoints, sync_by_event_rate, sync_from_pair
from evcam.io import EVENT_DTYPE
from evcam.sources import Clip


def burst_clip(fps=240.0, lead_s=1.0, bursts=((0.0, 0.15), (0.4, 0.1), (0.9, 0.3), (1.5, 0.05)), tail_s=1.0):
    """Tiny phone clip: still, then a pattern of brightness bursts of different lengths, then still."""
    dur = lead_s + 2.0 + tail_s
    t = np.arange(int(dur * fps)) / fps
    lum = np.full(len(t), 0.2)
    for t0, d in bursts:
        on = (t >= lead_s + t0) & (t < lead_s + t0 + d)
        lum[on] = 0.2 + 0.6 * np.sin(np.pi * (t[on] - lead_s - t0) / d) ** 2 * np.sin(2 * np.pi * 20 * t[on]) ** 2
    frames = np.repeat(lum[:, None, None], 16 * 16, 1).reshape(len(t), 16, 16).astype(np.float32)
    frames *= np.linspace(0.5, 1.5, 16)[None, None, :]
    return Clip(frames, np.round(t * 1e6).astype(np.int64), None, None, (16, 16), {})


def test_sync_recovers_known_offset():
    from evcam.simulate import simulate_events
    phone = burst_clip()
    ev = simulate_events(phone, c=0.15).events
    offset = 1.234
    rng = np.random.default_rng(0)
    noise = np.sort(rng.uniform(0, 8e6, 2000)).astype(np.int64)        # still lead-in and tail: sparse noise
    davis = np.empty(len(ev) + len(noise), EVENT_DTYPE)
    davis["t"] = np.concatenate([ev["t"] + int(offset * 1e6), noise])
    davis["x"] = davis["y"] = davis["p"] = 0
    davis = davis[np.argsort(davis["t"], kind="stable")]
    res = sync_by_event_rate(davis, phone, bin_ms=10, c=0.3)            # different threshold on the phone side
    assert abs(res.offset_s - offset) < 0.010
    assert res.reliable and res.score > 0.9


def test_sync_from_pair_and_check():
    assert sync_from_pair(1.4, 12.35) == pytest.approx(10.95)
    assert check_sync(None) is None and check_sync((1.4, 12.35)) == (1.4, 12.35)
    with pytest.raises(ValueError, match="SYNC must be None or"):
        check_sync(140)


def test_map_by_keypoints_recovers_similarity():
    rng = np.random.default_rng(1)
    kp = rng.uniform(0, 1000, (5, 13, 2))
    kp[0, 3] = kp[2, 7] = np.nan                                         # missing joints
    s, a, t = 0.31, np.deg2rad(4.0), np.array([20.0, -7.0])
    R = s * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    kd = kp @ R.T + t
    kd[1, 5] = np.nan
    A, rms = map_by_keypoints(kp, kd)
    np.testing.assert_allclose(A[:, :2], R, atol=1e-9)
    np.testing.assert_allclose(A[:, 2], t, atol=1e-6)
    assert rms < 1e-6
