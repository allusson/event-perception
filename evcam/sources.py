"""Data sources for notebooks 06 and 07: every loader returns the same `Clip` bundle.

A `Clip` holds whatever a source provides: frames (grayscale, linear intensity), their
timestamps and exposures, events, or several of these. Methods take a `Clip` and use the
parts they need, so adding a source means writing one loader and registering it in
`SOURCES`.

Gamma. Video files store gamma-encoded values (sRGB), but blur averages light in linear
intensity and the event model uses the log of linear intensity. `load_video` therefore
applies the inverse sRGB transfer function, and everything downstream works in linear
intensity. DAVIS APS frames (and REBlur, recorded with a DAVIS) are taken as linear.
"""

from __future__ import annotations

import dataclasses
import os
import warnings

import cv2
import numpy as np

from evcam.io import (EVENT_DTYPE, REBLUR_SHAPE, frames_look_accumulated, load_aedat4,  # noqa: F401
                      load_reblur_sample)

VIDEO_DIR = "data/video_data"      # phone videos
DAVIS_DIR = "data/davis_data"      # own DAVIS346 recordings


@dataclasses.dataclass
class Clip:
    """Common bundle returned by every loader.

    Attributes
    ----------
    frames : (N, H, W) float32, grayscale, linear intensity in [0, 1], or None
    frame_t : (N,) int64, time each frame represents, microseconds (for a real exposure,
        its midpoint). Empty if there are no frames.
    exposure : (N, 2) int64 exposure begin / end, microseconds, or None (frames rendered
        or captured at an instant, or exposure unknown)
    events : structured array in EVENT_DTYPE (x, y, t, p), t in microseconds on the same
        clock as frame_t, or None
    shape : (H, W)
    meta : dict, e.g. "source", "fps", "path", "c_true" (if events were simulated), "notes"
    """
    frames: np.ndarray | None
    frame_t: np.ndarray
    exposure: np.ndarray | None
    events: np.ndarray | None
    shape: tuple
    meta: dict = dataclasses.field(default_factory=dict)

    @property
    def n_frames(self):
        return 0 if self.frames is None else len(self.frames)

    @property
    def n_events(self):
        return 0 if self.events is None else len(self.events)

    def summary(self):
        """One-line description: source, frames, fps, duration, events."""
        parts = [f"source {self.meta.get('source', '?')}", f"{self.shape[1]}x{self.shape[0]}"]
        if self.n_frames:
            dur = (self.frame_t[-1] - self.frame_t[0]) / 1e6 if self.n_frames > 1 else 0.0
            parts.append(f"{self.n_frames} frames over {dur:.3f} s")
            if self.meta.get("fps"):
                parts.append(f"{self.meta['fps']:g} fps")
        else:
            parts.append("no frames")
        if self.n_events:
            t = self.events["t"]
            parts.append(f"{self.n_events:,} events over {(t[-1] - t[0]) / 1e6:.3f} s")
        else:
            parts.append("no events")
        return ", ".join(parts)


def ensure_data_dirs():
    """Create the (gitignored) folders for phone videos and own DAVIS recordings."""
    for d in (VIDEO_DIR, DAVIS_DIR):
        os.makedirs(d, exist_ok=True)


def srgb_to_linear(v):
    """Inverse sRGB transfer function, v in [0, 1] -> linear intensity in [0, 1]."""
    v = np.asarray(v, dtype=np.float32)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4).astype(np.float32)


# ---------------------------------------------------------------------------
# Synthetic video (default source)
# ---------------------------------------------------------------------------

def _texture(rng, h, w, n_shapes, spokes=0, up=4):
    """Random texture in [0.08, 0.9]: smooth noise plus filled rectangles and ellipses.

    Drawn at `up` times the size and downsampled with area averaging, so shape edges are
    anti-aliased. `spokes` > 0 adds that many dark radial wedges about the centre.
    """
    H, W = h * up, w * up
    noise = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 4.0 * up)
    img = 0.5 + 0.15 * noise / (noise.std() + 1e-9)
    for _ in range(n_shapes):
        val = float(rng.uniform(0.1, 0.9))
        cx, cy = int(rng.integers(0, W)), int(rng.integers(0, H))
        a, b = int(rng.integers(3 * up, max(4 * up, W // 5))), int(rng.integers(3 * up, max(4 * up, H // 5)))
        if rng.random() < 0.5:
            cv2.rectangle(img, (cx - a, cy - b), (cx + a, cy + b), val, -1)
        else:
            cv2.ellipse(img, (cx, cy), (a, b), float(rng.uniform(0, 180)), 0, 360, val, -1)
    if spokes:
        yy, xx = np.mgrid[0:H, 0:W]
        ang = np.arctan2(yy - (H - 1) / 2, xx - (W - 1) / 2)
        img[np.cos(spokes * ang) > 0.8] = 0.1
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return np.clip(img, 0.08, 0.9).astype(np.float32)


def synthetic_video(duration_s=0.5, fps=1000.0, shape=(260, 346), rev_per_s=2.0, square_velocity=(400.0, 0.0),
                    seed=0):
    """Procedurally rendered high-frame-rate grayscale video with known motion.

    A textured disk rotating about its centre and a textured square translating at
    constant velocity, on a static textured background. Linear intensity in [0.08, 0.9].
    Objects are composited with anti-aliased (sub-pixel) edges, and the textures are
    sampled bilinearly. Deterministic for a given seed.

    The rotation is inside the frame on purpose: the blur differs in direction and length
    across the disk, which a single motion per patch (CMax + Richardson-Lucy) cannot model.

    Parameters
    ----------
    duration_s, fps : length and frame rate; frame k is at time k / fps
    shape : (H, W)
    rev_per_s : disk rotation speed, revolutions per second (counter-clockwise on screen)
    square_velocity : (vx, vy) of the square, pixels per second
    seed : random seed for the textures

    Returns
    -------
    Clip with frames, frame_t (exposure None: each frame is an instant) and meta keys
    "disk_center", "disk_radius", "disk_box", "square_size", "square_xy0",
    "square_velocity", "rev_per_s"
    """
    rng = np.random.default_rng(seed)
    H, W = shape
    n = int(round(duration_s * fps))
    t_s = np.arange(n) / fps

    R = int(round(0.29 * H))
    cx, cy = 0.70 * W, 0.40 * H
    S = int(round(0.25 * H))
    sq0 = np.array([0.03 * W, H - S - 0.06 * H])
    vel = np.asarray(square_velocity, dtype=float)

    bg = _texture(rng, H, W, n_shapes=40)
    disk_tex = _texture(rng, 2 * R + 3, 2 * R + 3, n_shapes=25, spokes=6)
    sq_tex = _texture(rng, S, S, n_shapes=12)
    sq_tex[:3], sq_tex[-3:], sq_tex[:, :3], sq_tex[:, -3:] = 0.12, 0.12, 0.12, 0.12   # dark rims
    r_tex = np.hypot(*np.mgrid[0:2 * R + 3, 0:2 * R + 3] - (R + 1.0))
    disk_tex[(r_tex > R - 3.5) & (r_tex < R + 1)] = 0.12

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    disk_alpha = np.clip(R - np.hypot(xx - cx, yy - cy) + 0.5, 0, 1).astype(np.float32)
    ones = np.ones((S, S), np.float32)
    dc = (disk_tex.shape[1] - 1) / 2

    frames = np.empty((n, H, W), np.float32)
    for k, t in enumerate(t_s):
        img = bg.copy()
        ox, oy = sq0 + vel * t
        M = np.float32([[1, 0, ox], [0, 1, oy]])
        a = cv2.warpAffine(ones, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        tex = cv2.warpAffine(sq_tex, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        img = img * (1 - a) + tex * a

        th = 2 * np.pi * rev_per_s * t                       # counter-clockwise on screen (y down)
        cs, sn = np.cos(th), np.sin(th)
        # image point = Rot(-th) (texture point - dc) + (cx, cy), with y pointing down
        M = np.float32([[cs, sn, cx - cs * dc - sn * dc], [-sn, cs, cy + sn * dc - cs * dc]])
        tex = cv2.warpAffine(disk_tex, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        frames[k] = img * (1 - disk_alpha) + tex * disk_alpha

    meta = {"source": "synthetic", "fps": float(fps), "seed": seed, "rev_per_s": rev_per_s,
            "disk_center": (cx, cy), "disk_radius": R,
            "disk_box": (int(cx - R), int(cy - R), int(cx + R) + 1, int(cy + R) + 1),
            "square_size": S, "square_xy0": tuple(sq0), "square_velocity": tuple(vel),
            "notes": "rendered in linear intensity; frames are instants (no exposure)"}
    return Clip(frames=frames, frame_t=np.round(t_s * 1e6).astype(np.int64), exposure=None, events=None,
                shape=(H, W), meta=meta)


def square_box(clip, t_us):
    """(x0, y0, x1, y1) of the synthetic square at time t_us (from the clip's meta)."""
    m = clip.meta
    x0, y0 = np.asarray(m["square_xy0"]) + np.asarray(m["square_velocity"]) * t_us / 1e6
    return int(x0), int(y0), int(x0) + m["square_size"] + 1, int(y0) + m["square_size"] + 1


# ---------------------------------------------------------------------------
# Video files (e.g. iPhone slow motion)
# ---------------------------------------------------------------------------

FFMPEG_HINT = "ffmpeg -i input.MOV -c:v libx264 -crf 18 output.mp4"


def load_video(path, capture_fps=None, t_start_s=0.0, duration_s=None, resize_to=(260, 346), linearize=True):
    """Read a video file as a grayscale, linear-intensity `Clip`.

    Timestamps come from `capture_fps` when it is given, not from the container frame
    rate: iPhone slow-motion files are often stored or exported at a playback rate (e.g.
    30 fps) while the frames were captured at 240 fps, and the container rate would make
    every time 8x too long. Both rates are printed.

    Each frame is converted to float, linearized (inverse sRGB, if `linearize`), centre-
    cropped to the aspect ratio of `resize_to`, resized with area averaging and converted
    to grayscale with the Rec. 709 luminance weights. Cropping, resizing and the grayscale
    mix happen after linearization, because they average light.

    Parameters
    ----------
    path : video file readable by OpenCV
    capture_fps : true capture frame rate; None uses the container rate
    t_start_s, duration_s : window in capture time, seconds (duration None: to the end)
    resize_to : (H, W) of the output, or None to keep the native size
    linearize : apply the inverse sRGB transfer function

    Returns
    -------
    Clip with frames and frame_t (starting at 0 at the first returned frame), exposure None
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read() if cap.isOpened() else (False, None)
    if not ok:
        cap.release()
        raise RuntimeError(f"OpenCV cannot decode {path} (iPhone HEVC is not supported by every OpenCV build). "
                           f"Convert it first: {FFMPEG_HINT}")
    container_fps = float(cap.get(cv2.CAP_PROP_FPS)) or None
    fps = float(capture_fps) if capture_fps else container_fps
    if not fps:
        cap.release()
        raise ValueError(f"{path}: the container has no frame rate; pass capture_fps")
    print(f"{os.path.basename(path)}: container fps {container_fps}, capture fps used for timestamps {fps:g}")

    k0 = int(round(t_start_s * fps))
    n_max = None if duration_s is None else int(round(duration_s * fps))
    frames, k = [], 0
    while ok:
        if k >= k0:
            frames.append(_video_frame(frame, resize_to, linearize))
            if n_max is not None and len(frames) >= n_max:
                break
        ok, frame = cap.read()
        k += 1
    cap.release()
    if not frames:
        raise ValueError(f"{path}: no frames after t_start_s = {t_start_s}")
    frames = np.stack(frames)
    frame_t = np.round(np.arange(len(frames)) / fps * 1e6).astype(np.int64)
    meta = {"source": "video", "path": path, "fps": fps, "container_fps": container_fps,
            "t_start_s": t_start_s, "linearized": linearize,
            "notes": "exposure unknown; frame_t from capture_fps" if capture_fps else "exposure unknown"}
    return Clip(frames=frames, frame_t=frame_t, exposure=None, events=None, shape=frames.shape[1:], meta=meta)


def _video_frame(bgr, resize_to, linearize):
    img = bgr.astype(np.float32) / (65535.0 if bgr.dtype == np.uint16 else 255.0)
    if img.ndim == 2:
        img = img[..., None]
    if linearize:
        img = srgb_to_linear(img)
    if resize_to is not None:
        H, W = resize_to
        h, w = img.shape[:2]
        if h * W > w * H:                       # too tall: crop rows
            hh = int(round(w * H / W)); y0 = (h - hh) // 2
            img = img[y0:y0 + hh]
        else:                                   # too wide: crop columns
            ww = int(round(h * W / H)); x0 = (w - ww) // 2
            img = img[:, x0:x0 + ww]
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        if img.ndim == 2:
            img = img[..., None]
    if img.shape[2] == 1:
        return np.ascontiguousarray(img[..., 0])
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    return (0.0722 * b + 0.7152 * g + 0.2126 * r).astype(np.float32)


# ---------------------------------------------------------------------------
# DAVIS346 recordings
# ---------------------------------------------------------------------------

def load_davis(path, t_start_s=0.0, duration_s=None):
    """Load a DV .aedat4 recording as a `Clip` (events, and APS frames if they are real).

    Wraps `evcam.io.load_aedat4`. If `frames_look_accumulated` flags the frames as built
    from events by DV's Accumulator, `frames` is None and the reason is in meta["notes"].
    Times are shifted so the window start is 0; frames are kept if their whole exposure
    lies inside the window. APS values are divided by 255 and taken as linear.

    Parameters
    ----------
    path : .aedat4 file
    t_start_s, duration_s : window, seconds from the first event (duration None: to the end)

    Returns
    -------
    Clip with events, and frames / frame_t (exposure midpoints) / exposure when available
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        events, frames, sensor_size = load_aedat4(path)
    shape = (sensor_size[1], sensor_size[0])
    t0 = int(round(t_start_s * 1e6))
    t1 = int(events["t"][-1]) + 1 if duration_s is None else t0 + int(round(duration_s * 1e6))
    ev = events[(events["t"] >= t0) & (events["t"] < t1)].copy()
    ev["t"] -= t0

    flagged, why = frames_look_accumulated(frames)
    meta = {"source": "davis", "path": path, "t_start_s": t_start_s}
    if not frames:
        meta["notes"] = "no frames in the file"
    elif flagged:
        meta["notes"] = f"frames dropped: they look like DV Accumulator output ({why['summary']})"
    frames = [f for f in frames if f["exposure_begin_t"] >= t0 and f["exposure_end_t"] <= t1]
    if flagged or not frames:
        meta.setdefault("notes", "no frame exposure inside the window")
        return Clip(None, np.zeros(0, np.int64), None, ev, shape, meta)

    exposure = np.array([[f["exposure_begin_t"], f["exposure_end_t"]] for f in frames], np.int64) - t0
    imgs = np.stack([np.asarray(f["image"], np.float32) / 255.0 for f in frames])
    meta["fps"] = 1e6 / float(np.median(np.diff([f["t"] for f in frames]))) if len(frames) > 1 else None
    meta["notes"] = "APS frames, taken as linear"
    return Clip(imgs, exposure.mean(axis=1).round().astype(np.int64), exposure, ev, shape, meta)


# ---------------------------------------------------------------------------
# REBlur (Sun et al., EFNet, ECCV 2022)
# ---------------------------------------------------------------------------

def load_reblur_clip(seq, idx, split="test", root="data/reblur"):
    """One REBlur sample as a `Clip`: the blurry frame, its exposure and events.

    Wraps `evcam.io.load_reblur_sample`. Times are on the REBlur clock.

    Returns
    -------
    Clip with one frame (the blurry APS frame), its exposure, the events inside it, and
    meta "sharp" (ground truth, (H, W) float32 or None), "scer" and "mask" (as stored)
    """
    s = load_reblur_sample(seq, idx, split=split, root=root)
    exposure = np.array([[s["t_begin"], s["t_end"]]], np.int64)
    sharp = None if s["sharp"] is None else s["sharp"].astype(np.float32)
    meta = {"source": "reblur", "sequence": seq, "index": idx, "split": split, "sharp": sharp,
            "scer": s["scer"], "mask": s["mask"], "notes": "DAVIS APS frame, taken as linear"}
    return Clip(s["blurry"].astype(np.float32)[None], exposure.mean(axis=1).round().astype(np.int64), exposure,
                s["events"], REBLUR_SHAPE, meta)


# ---------------------------------------------------------------------------
# HS-ERGB (Tulyakov et al., TimeLens, CVPR 2021): not implemented
# ---------------------------------------------------------------------------

def load_hs_ergb(sequence, root="data/hs_ergb", t_start_s=0.0, duration_s=None):
    """Not implemented. HS-ERGB pairs a Prophesee Gen4 event camera with a high-speed RGB camera.

    It should return a `Clip` with the RGB frames converted to grayscale linear intensity
    (frames, frame_t from the sequence's image timestamps file, exposure from its exposure
    times if given, else None), the events in EVENT_DTYPE with t in microseconds on the
    frame clock (p mapped to {0, 1}), shape = the event sensor size after the dataset's
    alignment of the two cameras, and meta "source": "hs_ergb", "fps", "path".
    """
    raise NotImplementedError("HS-ERGB loader is a stub; see its docstring for what it should return.")


SOURCES = {
    "synthetic": synthetic_video,
    "video": load_video,
    "davis": load_davis,
    "reblur": load_reblur_clip,
}
