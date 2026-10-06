"""EDI on a synthetic 1D case with a known answer.

A step edge moves right at constant speed across a single row of pixels. Everything is
analytic: each pixel sees one intensity step at the moment the edge crosses its centre, so
the blurry value is a time-weighted mix of the two levels and the ideal event camera emits
exactly N events of one polarity at the crossing time, where N * c = log(hi / lo).
"""

import warnings

import numpy as np
import pytest

from evcam.deblur import edi_deblur, edi_divisor, edi_select_c
from evcam.io import EVENT_DTYPE

W = 64
T0, T1 = 1_000_000, 1_030_000          # 30 ms exposure, microseconds
C_TRUE, N_PER_STEP = 0.3, 3            # 3 events per crossing -> hi / lo = exp(0.9)
LO = 0.2
HI = LO * np.exp(C_TRUE * N_PER_STEP)
X_START, SPEED = 20.0, 800.0           # edge position at T0 (px) and speed (px/s): moves 24 px


def edge_pos(t):
    return X_START + SPEED * (t - T0) / 1e6


def sharp_at(t):
    """HI to the left of the edge (already passed), LO to the right."""
    return np.where(np.arange(W) < edge_pos(t), HI, LO)[None, :]


def make_case():
    x = np.arange(W)
    t_cross = T0 + (x - X_START) / SPEED * 1e6          # when the edge reaches pixel x
    crossed = (t_cross >= T0) & (t_cross < T1)
    # time-average of the per-pixel step: LO before the crossing, HI after
    frac_hi = np.clip((T1 - t_cross) / (T1 - T0), 0, 1)
    blurry = (frac_hi * HI + (1 - frac_hi) * LO)[None, :]

    ev = np.zeros(crossed.sum() * N_PER_STEP, dtype=EVENT_DTYPE)
    ev["x"] = np.repeat(x[crossed], N_PER_STEP)
    ev["t"] = np.repeat(np.round(t_cross[crossed]).astype(np.int64), N_PER_STEP)
    ev["p"] = 1                                          # LO -> HI is a brightness increase
    return blurry, ev[np.argsort(ev["t"], kind="stable")]


def test_edi_recovers_step_edge_at_true_c():
    blurry, ev = make_case()
    t_ref = (T0 + T1) / 2
    truth = sharp_at(t_ref)
    out = edi_deblur(blurry, ev, T0, T1, C_TRUE)
    assert out.shape == blurry.shape
    assert np.abs(blurry - truth).max() > 0.1            # the input really is blurred
    assert np.abs(out - truth).max() < 0.01              # and EDI undoes it
    # a wrong threshold must be clearly worse on both sides
    for c_bad in (0.15, 0.45):
        assert np.abs(edi_deblur(blurry, ev, T0, T1, c_bad) - truth).max() > 0.05


def test_edi_reference_time():
    """t_ref picks which instant is reconstructed."""
    blurry, ev = make_case()
    for t_ref in (T0, T0 + 10_000, T1):
        out = edi_deblur(blurry, ev, T0, T1, C_TRUE, t_ref=t_ref)
        assert np.abs(out - sharp_at(t_ref)).max() < 0.01


def test_binning_converges():
    blurry, ev = make_case()
    truth = sharp_at((T0 + T1) / 2)
    err = [np.abs(edi_deblur(blurry, ev, T0, T1, C_TRUE, n_bins=n) - truth).max() for n in (4, 16, 60)]
    assert err[0] > err[1] > err[2]


@pytest.mark.xfail(reason="ideal 1D step is a degenerate case for the c criterion")
def test_select_c_near_truth():
    # Expected to fail, kept as documentation. A blurred monotone step has exactly the same
    # total variation as the sharp step, so the TV term is flat for every c below the true
    # one, and with noise-free ideal events nothing penalizes under-correction. The
    # criterion is meant for real images (texture, noise, many edges) and its weighting is
    # tuned on the REBlur train split, not on this toy case.
    blurry, ev = make_case()
    c, curve = edi_select_c(blurry, ev, T0, T1)
    assert abs(c - C_TRUE) < 0.05
    assert len(curve["c"]) == len(curve["objective"]) > 12
    assert np.all(np.diff(curve["c"]) > 0)


def test_no_events_returns_input_with_warning():
    blurry, _ = make_case()
    with pytest.warns(UserWarning, match="no events"):
        out = edi_deblur(blurry, np.zeros(0, dtype=EVENT_DTYPE), T0, T1, C_TRUE)
    np.testing.assert_allclose(out, blurry)


def test_partial_coverage_warns_and_runs():
    """A stream that starts after the exposure begins (like the first DAVIS frame)."""
    blurry, ev = make_case()
    late = np.zeros(1, dtype=EVENT_DTYPE); late["t"] = T1 + 5_000
    stream = np.concatenate([ev[ev["t"] > T0 + 8_000], late])
    with pytest.warns(UserWarning, match="covers only part"):
        d = edi_divisor(stream, T0, T1, (T0 + T1) / 2, C_TRUE, blurry.shape)
    assert np.isfinite(d).all()
    # events cropped to the window already are not a "partial stream": no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        edi_divisor(ev, T0, T1, (T0 + T1) / 2, C_TRUE, blurry.shape)
