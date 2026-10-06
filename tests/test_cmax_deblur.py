import numpy as np

from evcam.deblur import cmax_deblur, linear_psf
from evcam.io import EVENT_DTYPE


def test_linear_psf():
    psf = linear_psf((6.0, 0.0))
    assert psf.shape[0] % 2 == 1 and abs(psf.sum() - 1) < 1e-12
    row = psf[psf.shape[0] // 2]
    assert np.isclose(row.sum(), 1) and (row > 0.01).sum() in (6, 7)     # a horizontal 6 px line
    np.testing.assert_allclose(psf, psf[::-1, ::-1], atol=1e-12)          # centred


def test_no_events_returns_input():
    """The Hann tiles must sum to 1 everywhere, borders included."""
    img = np.random.default_rng(0).random((100, 150))
    for gate in (True, False):
        out = cmax_deblur(img, np.zeros(0, dtype=EVENT_DTYPE), 0, 1000, event_gate=gate)
        np.testing.assert_allclose(out, img, atol=1e-12)
