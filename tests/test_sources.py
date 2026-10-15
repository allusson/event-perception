"""Source loaders: the synthetic video and video files read through OpenCV."""

import cv2
import numpy as np
import pytest

from evcam.sources import load_video, srgb_to_linear, synthetic_video


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
