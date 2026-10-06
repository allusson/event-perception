"""Event representations that are not already provided by tonic."""

import numpy as np


def time_surface(events, sensor_size, tau=5000.0):
    """Exponentially decaying time surface, separately for ON and OFF events.

    The reference time is the last event in `events`; each pixel holds
    exp(-(t_ref - t_last) / tau) for the most recent event at that pixel (0 if none).

    Parameters
    ----------
    events : structured array (x, y, t, p), t in microseconds, p in {0, 1}
    sensor_size : (W, H, 2)
    tau : decay constant in microseconds

    Returns
    -------
    ts_on, ts_off : two (H, W) float arrays with values in [0, 1]
    """
    W, H, _ = sensor_size
    t_ref = events["t"][-1]  # reference time: last event in the sample
    decay = np.exp(-(t_ref - events["t"]) / tau)
    surfaces = []
    for pol in (1, 0):
        m = events["p"] == pol
        ts = np.zeros((H, W))
        np.maximum.at(ts, (events["y"][m], events["x"][m]), decay[m])
        surfaces.append(ts)
    return surfaces[0], surfaces[1]


def event_image(events, sensor_size):
    """Accumulate events into one signed count image (ON count - OFF count per pixel).

    Parameters
    ----------
    events : structured array (x, y, t, p), p in {0, 1}
    sensor_size : (W, H, 2)

    Returns
    -------
    (H, W) float array
    """
    ev_img = np.zeros((sensor_size[1], sensor_size[0]))
    np.add.at(ev_img, (events["y"], events["x"]), np.where(events["p"] == 1, 1, -1))
    return ev_img
