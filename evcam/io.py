"""Loading event data: DV .aedat4 recordings (DAVIS346) and DSEC sequences."""

import glob
import warnings

import aedat
import cv2
import h5py
import hdf5plugin  # noqa: F401  (registers the Blosc/ZSTD filters DSEC event files need)
import numpy as np

EVENT_DTYPE = np.dtype([("x", "<i8"), ("y", "<i8"), ("t", "<i8"), ("p", "<i8")])  # same as tonic's N-MNIST


# ---------------------------------------------------------------------------
# DV .aedat4 recordings
# ---------------------------------------------------------------------------

def load_aedat4(path):
    """Load a DV .aedat4 recording.

    Timestamps in the file are absolute (microseconds since the Unix epoch). Everything
    returned here is shifted onto one clock that starts at 0 at the first event.

    Parameters
    ----------
    path : str, path to the .aedat4 file

    Returns
    -------
    events : structured array of shape (N,) with fields (x, y, t, p),
        t in microseconds starting at 0, p in {0, 1}
    frames : list of dicts, one per frame in the file, with keys
        "t"                : frame timestamp, microseconds
        "exposure_begin_t" : start of the exposure, microseconds
        "exposure_end_t"   : end of the exposure, microseconds
        "image"            : (H, W) grayscale image
        All three times are on the same zeroed clock as the events.
        These are whatever frame stream DV was told to record. That is the camera's APS
        frames only if the capture node's `frames` output was wired to the file output;
        if the Accumulator module was wired in instead, they are images rendered from the
        events. `frames_look_accumulated` tells the two apart, and a warning is raised
        here when the frames look event-built.
    sensor_size : (W, H, 2)
    """
    decoder = aedat.Decoder(path)
    streams = decoder.id_to_stream()
    ev_info = next(s for s in streams.values() if s["type"] == "events")
    sensor_size = (ev_info["width"], ev_info["height"], 2)

    ev_chunks, raw_frames = [], []
    for packet in decoder:
        if "events" in packet:
            ev_chunks.append(packet["events"])
        elif "frame" in packet:
            f = packet["frame"]
            raw_frames.append((int(f["t"]), int(f["exposure_begin_t"]), int(f["exposure_end_t"]),
                               f["pixels"].squeeze()))

    raw = np.concatenate(ev_chunks)
    t0 = int(raw["t"][0])

    events = np.empty(len(raw), dtype=EVENT_DTYPE)
    events["x"] = raw["x"]
    events["y"] = raw["y"]
    events["t"] = raw["t"].astype(np.int64) - t0
    events["p"] = raw["on"].astype(np.int64)

    frames = [{"t": t - t0, "exposure_begin_t": tb - t0, "exposure_end_t": te - t0, "image": img}
              for t, tb, te, img in raw_frames]
    flagged, why = frames_look_accumulated(frames)
    if flagged:
        warnings.warn(f"{path}: the frames look like DV Accumulator output (rendered from events), "
                      f"not APS sensor frames ({why['summary']}). They are fine to look at, but "
                      "frame-based deblurring (EDI, EFNet) is not meaningful on them.", stacklevel=2)
    return events, frames, sensor_size


def frames_look_accumulated(frames, n_check=20):
    """Heuristic: are these frames DV Accumulator output (built from events) rather than APS?

    Two signatures, either of which flags the recording:
    - Quantized histogram. The Accumulator adds a fixed step per event (default 0.15 of
      full scale, i.e. 38 grey levels) to a flat background, so a few exact grey values
      hold most of the non-saturated pixels. Photon and read noise spread a real sensor
      frame over many levels.
    - Exposure equal to the frame period. The Accumulator stamps each image with its
      accumulation window, which is exactly the time between frames. A real shutter is
      shorter than the frame period.

    Parameters
    ----------
    frames : list of frame dicts from `load_aedat4`
    n_check : number of frames (evenly spaced) used for the histogram test

    Returns
    -------
    flagged : bool
    info : dict with "quantized" (bool), "top3_fraction" (median share of non-saturated
        pixels held by the 3 most common grey values), "exposure_equals_period" (bool),
        "exposure_us", "period_us" and a one-line "summary"
    """
    if len(frames) < 2:
        return False, {"quantized": False, "top3_fraction": 0.0, "exposure_equals_period": False,
                       "exposure_us": None, "period_us": None, "summary": "too few frames to tell"}
    fracs = []
    for i in np.linspace(0, len(frames) - 1, min(n_check, len(frames))).astype(int):
        img = np.asarray(frames[i]["image"])
        hist = np.bincount(img[(img > img.min()) & (img < img.max())].ravel().astype(np.int64))
        if hist.sum():
            fracs.append(np.sort(hist)[-3:].sum() / hist.sum())
    top3 = float(np.median(fracs)) if fracs else 0.0
    quantized = top3 > 0.25      # real 8-bit sensor frames sit around 0.03-0.10

    exposure = float(np.median([f["exposure_end_t"] - f["exposure_begin_t"] for f in frames]))
    period = float(np.median(np.diff([f["t"] for f in frames])))
    same = exposure > 0 and exposure == period

    parts = []
    if quantized:
        parts.append(f"{top3:.0%} of non-saturated pixels sit on 3 grey values")
    if same:
        parts.append(f"exposure equals the frame period, {exposure / 1e3:.1f} ms")
    return quantized or same, {"quantized": quantized, "top3_fraction": top3,
                               "exposure_equals_period": same, "exposure_us": exposure,
                               "period_us": period, "summary": "; ".join(parts) or "looks like APS"}


def crop_time(events, t_start_us, duration_us):
    """Keep events in [t_start, t_start + duration) and re-zero their timestamps.

    Parameters
    ----------
    events : structured array (x, y, t, p), t in microseconds
    t_start_us, duration_us : window start and length, microseconds

    Returns
    -------
    structured array (x, y, t, p) with t starting at 0 at the window start
    """
    m = (events["t"] >= t_start_us) & (events["t"] < t_start_us + duration_us)
    out = events[m].copy()
    out["t"] -= t_start_us
    return out


# ---------------------------------------------------------------------------
# DSEC
# ---------------------------------------------------------------------------

def open_dsec_events(events_path):
    """Open a DSEC `events.h5` file (left open so time windows can be sliced lazily).

    Returns
    -------
    h5py.File with datasets `events/x`, `events/y`, `events/t` (microseconds),
    `events/p` (0/1), `t_offset` (microseconds) and `ms_to_idx`.
    """
    return h5py.File(events_path, "r")


def load_rectify_map(rectify_map_path):
    """Load a DSEC `rectify_map.h5`.

    Returns
    -------
    rect_map : (H, W, 2) float array; rect_map[y, x] = (x_rect, y_rect), the rectified
        pixel coordinates of raw sensor pixel (x, y)
    """
    with h5py.File(rectify_map_path, "r") as f:
        return f["rectify_map"][:]


def load_flow_timestamps(timestamps_path):
    """Load a DSEC `*_optical_flow_forward_timestamps.txt` file.

    Returns
    -------
    (K, 2) int64 array of (t_start, t_end) per ground-truth flow map, in microseconds on
    the global DSEC clock
    """
    return np.loadtxt(timestamps_path, delimiter=",", dtype=np.int64)


def list_flow_files(flow_dir):
    """Sorted list of ground-truth flow PNG paths in `flow_dir`.

    The sorted files line up row by row with the timestamp file.
    """
    return sorted(glob.glob(f"{flow_dir}/*.png"))


def load_flow(path):
    """Decode one DSEC ground-truth flow PNG.

    Returns
    -------
    flow : (H, W, 2) float64, (x, y) displacement in pixels over the flow window
    valid : (H, W) bool, True where ground truth exists
    """
    im = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)[..., ::-1].astype(np.float64)
    flow = np.stack([(im[..., 0] - 2**15) / 128, (im[..., 1] - 2**15) / 128], axis=-1)
    return flow, im[..., 2] > 0


def get_events(ev, rect_map, t0_abs, t1_abs):
    """Read the DSEC events in [t0_abs, t1_abs) and rectify their coordinates.

    Parameters
    ----------
    ev : open events file from `open_dsec_events`
    rect_map : (H, W, 2) rectification map from `load_rectify_map`
    t0_abs, t1_abs : window start and end, microseconds on the global DSEC clock
        (the clock used by the flow timestamps)

    Returns
    -------
    x, y : (N,) float, rectified pixel coordinates
    tn : (N,) float, time normalized to [0, 1) over the window
    p : (N,) polarity in {0, 1}
    """
    t_offset = int(ev["t_offset"][()])
    ms_to_idx = ev["ms_to_idx"][:]

    t0, t1 = t0_abs - t_offset, t1_abs - t_offset
    i0 = ms_to_idx[t0 // 1000]
    i1 = ms_to_idx[min(t1 // 1000 + 1, len(ms_to_idx) - 1)]
    t = ev["events/t"][i0:i1].astype(np.int64)
    m = (t >= t0) & (t < t1)
    x, y, p = ev["events/x"][i0:i1][m], ev["events/y"][i0:i1][m], ev["events/p"][i0:i1][m]
    xy = rect_map[y, x]
    return xy[:, 0], xy[:, 1], (t[m] - t0) / (t1 - t0), p
