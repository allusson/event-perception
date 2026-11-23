"""Event image metrics on tiny images with known answers."""

import numpy as np

from evcam.metrics import chamfer_distance, motion_selectivity, moving_segments, tolerant_prf


def test_chamfer_shifted_line():
    a = np.zeros((20, 20), bool); a[5, 2:18] = True
    b = np.zeros((20, 20), bool); b[8, 2:18] = True      # same line 3 px lower
    assert chamfer_distance(a, a) == 0
    assert np.isclose(chamfer_distance(a, b), 3.0)
    assert np.isnan(chamfer_distance(a, np.zeros_like(a)))


def test_tolerant_prf():
    ref = np.zeros((20, 20), bool); ref[5, 0:10] = True               # 10 px
    pred = np.zeros((20, 20), bool); pred[7, 0:10] = True; pred[15, 0:10] = True   # 10 px 2 away, 10 px 10 away
    p, r, f = tolerant_prf(pred, ref, tol=3)
    assert np.isclose(p, 0.5) and np.isclose(r, 1.0) and np.isclose(f, 2 * 0.5 / 1.5)
    assert tolerant_prf(pred, ref, tol=1)[2] == 0.0


def test_motion_selectivity():
    # two horizontal segments: joints 0-1 move, 2-3 stay
    xy = np.array([[[10, 10], [40, 10], [10, 40], [40, 40]],
                   [[10, 14], [40, 14], [10, 40], [40, 40]]], float)
    moving, still = moving_segments(xy, 1, [(0, 1), (2, 3)], move_px=2)
    assert moving == [(0, 1)] and still == [(2, 3)]
    ev = np.zeros((60, 60), bool)
    ev[14, 10:41] = True          # a line of events on the moving segment
    ev[40, 10:41:4] = True        # a quarter as dense on the still one
    ratio = motion_selectivity(ev, xy[1], moving, still, radius=3)
    assert 3.5 < ratio < 4.5
