import numpy as np

from evcam.cmax import cmax_rotation, iwe, iwe_from_coords, rotation_warp
from evcam.io import rotation_flow


def _rot(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.cross(np.eye(3), axis)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


# pixel -> direction: pinhole (f = 200 px, centre (120, 90)) followed by a fixed rotation
M = _rot((1, 2, 3), 0.7) @ np.linalg.inv(np.array([[200.0, 0, 120], [0, 200, 90], [0, 0, 1]]))
OMEGA = _rot((1, 2, 3), 0.7) @ (np.radians(4.0) * np.array([np.sin(np.radians(11)), 0, np.cos(np.radians(11))]))


def test_iwe_from_coords_matches_iwe():
    rng = np.random.default_rng(0)
    x, y, tn = rng.uniform(0, 100, 500), rng.uniform(0, 80, 500), rng.random(500)
    d = (5.0, -3.0)
    np.testing.assert_allclose(iwe_from_coords(x - d[0] * tn, y - d[1] * tn, (80, 100)), iwe(x, y, tn, d, (80, 100)))


def test_rotation_warp_matches_rotation_flow_for_small_times():
    x, y = np.array([10.0, 120.0, 230.0]), np.array([20.0, 90.0, 170.0])
    dt = 1e-4
    xw, yw = rotation_warp(x, y, np.full(3, dt), OMEGA, M, 0.0)
    v = np.stack([x - xw, y - yw], axis=1) / dt
    np.testing.assert_allclose(v, rotation_flow(M, OMEGA, x, y), rtol=1e-3, atol=1e-3)


def test_cmax_rotation_recovers_omega():
    rng = np.random.default_rng(0)
    H, W, T, pad = 180, 240, 2.0, 50
    n_stars, per_star = 150, 100
    x0, y0 = rng.uniform(0, W, n_stars), rng.uniform(0, H, n_stars)
    t = rng.uniform(0, T, (n_stars, per_star)).ravel()
    # stars at t_ref = 0 moved forward under OMEGA: rotation_warp with -OMEGA
    x, y = rotation_warp(np.repeat(x0, per_star), np.repeat(y0, per_star), t, -OMEGA, M, 0.0)
    x, y = x + rng.normal(0, 0.3, x.size), y + rng.normal(0, 0.3, y.size)
    m = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    x, y, t = x[m], y[m], t[m]

    omega, c = cmax_rotation(x, y, t, M, 0.0, (H + 2 * pad, W + 2 * pad), (-pad, -pad))
    assert abs(np.linalg.norm(omega) - np.linalg.norm(OMEGA)) / np.linalg.norm(OMEGA) < 0.05
    cos = omega @ OMEGA / (np.linalg.norm(omega) * np.linalg.norm(OMEGA))
    assert np.degrees(np.arccos(np.clip(cos, -1, 1))) < 3.0
    assert c > 0
