"""Source loaders: the synthetic video and video files read through OpenCV."""

import cv2
import numpy as np
import pytest

from evcam.sources import frame_indices_from_timestamps, load_video, srgb_to_linear, synthetic_video


def test_synthetic_video_deterministic_shape_and_timestamps():
    a = synthetic_video(duration_s=0.02, fps=500, shape=(60, 80), seed=3)
    b = synthetic_video(duration_s=0.02, fps=500, shape=(60, 80), seed=3)
    assert a.frames.shape == (10, 60, 80) and a.frames.dtype == np.float32
    assert a.shape == (60, 80)
    np.testing.assert_array_equal(a.frames, b.frames)
    np.testing.assert_array_equal(a.frame_t, np.arange(10) * 2000)
    assert a.frame_t.dtype == np.int64
    assert 0 <= a.frames.min() and a.frames.max() <= 1
    assert not np.array_equal(a.frames[0], a.frames[-1])           # something moves
    assert not np.array_equal(a.frames, synthetic_video(duration_s=0.02, fps=500, shape=(60, 80), seed=4).frames)


@pytest.fixture
def mp4(tmp_path):
    """12 frames of a uniform grey ramp written at a 30 fps container rate."""
    path = str(tmp_path / "ramp.mp4")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    if not w.isOpened():
        pytest.skip("OpenCV has no mp4v encoder")
    for k in range(12):
        w.write(np.full((48, 64, 3), 40 + 15 * k, np.uint8))
    w.release()
    return path


def test_load_video_container_fps(mp4):
    clip = load_video(mp4, resize_to=(24, 32))
    assert clip.frames.shape == (12, 24, 32)
    np.testing.assert_allclose(clip.frame_t, np.round(np.arange(12) / 30 * 1e6))
    # linearized: grey level v maps to srgb_to_linear(v / 255) (codec error a few levels)
    np.testing.assert_allclose(clip.frames[:, 12, 16], srgb_to_linear((40 + 15 * np.arange(12)) / 255), atol=0.02)


def test_load_video_capture_fps_overrides_timestamps(mp4):
    clip = load_video(mp4, capture_fps=240, t_start_s=2 / 240, duration_s=5 / 240, resize_to=None)
    assert clip.frames.shape == (5, 48, 64)
    np.testing.assert_array_equal(clip.frame_t, np.round(np.arange(5) / 240 * 1e6))
    assert clip.meta["fps"] == 240 and clip.meta["container_fps"] == pytest.approx(30)
    # the window starts at frame 2
    np.testing.assert_allclose(clip.frames[0].mean(), srgb_to_linear(70 / 255), atol=0.02)


def test_load_video_unreadable(tmp_path):
    bad = tmp_path / "bad.mov"
    bad.write_bytes(b"not a video")
    with pytest.raises(RuntimeError, match="ffmpeg -i input.MOV"):
        load_video(str(bad))


def test_frame_indices_uniform_timestamps():
    stamps = np.arange(20) * 1000 / 30                       # 30 fps timeline
    index, n_missing, step = frame_indices_from_timestamps(stamps)
    np.testing.assert_array_equal(index, np.arange(20))
    assert n_missing == 0 and step == pytest.approx(1000 / 30)
    # frame_t as load_video computes it equals the old uniform grid
    np.testing.assert_array_equal(np.round(index / 240 * 1e6).astype(np.int64),
                                  np.round(np.arange(20) / 240 * 1e6).astype(np.int64))


def test_frame_indices_missing_frames():
    # 240 fps capture on a 60 fps timeline, stored in 1/600 s ticks; frames 2, 5, 6 missing,
    # and one step quantized a tick long (18.333 ms) as in iPhone slow-motion files
    captured = np.array([0, 1, 3, 4, 7, 8, 9])
    stamps = np.round(captured * 10) / 0.6                   # ms, multiples of 1/600 s
    stamps[5:] += 1000 / 600                                 # 8 - 7 shows up as 18.333 ms
    index, n_missing, step = frame_indices_from_timestamps(stamps)
    np.testing.assert_array_equal(index, captured)
    assert n_missing == 3 and step == pytest.approx(1000 / 60)
    np.testing.assert_array_equal(np.round(index / 240 * 1e6), np.round(captured / 240 * 1e6))


def test_frame_indices_invalid_steps():
    stamps = np.cumsum([0] + [10.0, 15.0] * 10)              # 1.5x steps: not whole multiples
    with pytest.raises(ValueError, match="clip.mov.*10.000 to 15.000 ms"):
        frame_indices_from_timestamps(stamps, "clip.mov")
    with pytest.raises(ValueError, match="not increasing"):
        frame_indices_from_timestamps([0, 10, 10, 20], "clip.mov")
