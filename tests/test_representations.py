import numpy as np

from evcam.io import EVENT_DTYPE
from evcam.representations import scer, scer_mask


def test_scer_bins_sign_and_order():
    """One ON event per time slice at its own pixel: check which channels each one lands in."""
    ev = np.zeros(6, dtype=EVENT_DTYPE)
    ev["x"] = np.arange(6)                    # event k sits at pixel (0, k)
    ev["t"] = 100 + 10 * np.arange(6) + 5     # middle of slice k of the window [100, 160)
    ev["p"] = 1
    v = scer(ev, 100, 160, (1, 6))
    expected = np.array([[-1, -1, -1, 0, 0, 0],    # ch 0: whole first half, negated
                         [0, -1, -1, 0, 0, 0],     # ch 1
                         [0, 0, -1, 0, 0, 0],      # ch 2: slice just before the midpoint
                         [0, 0, 0, 1, 0, 0],       # ch 3: slice just after the midpoint
                         [0, 0, 0, 1, 1, 0],       # ch 4
                         [0, 0, 0, 1, 1, 1]])      # ch 5: whole second half
    np.testing.assert_array_equal(v[:, 0, :], expected)


def test_scer_ignores_events_outside_window_and_off_polarity_is_negative():
    ev = np.zeros(3, dtype=EVENT_DTYPE)
    ev["t"] = [50, 150, 500]
    ev["p"] = [1, 0, 1]
    v = scer(ev, 100, 160, (8, 8))
    assert v[5, 0, 0] == -1 and np.abs(v).sum() == 1   # only the OFF event at t=150, in the last slice (channel 5)
    assert scer_mask(v)[0, 0] == 1 and scer_mask(v).sum() == 1
