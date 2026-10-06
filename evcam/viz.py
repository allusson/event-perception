"""Visualization helpers shared across notebooks."""

import cv2
import numpy as np


def flow_to_rgb(flow, valid=None, max_mag=None):
    """Standard HSV flow encoding: hue = direction, brightness = magnitude.

    Parameters
    ----------
    flow : (H, W, 2) flow in pixels, no NaNs
    valid : optional (H, W) bool mask; pixels outside it are drawn black
    max_mag : magnitude in pixels mapped to full brightness; defaults to the 99th
        percentile of the (valid) magnitudes

    Returns
    -------
    (H, W, 3) uint8 RGB image
    """
    mag, ang = cv2.cartToPolar(flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32))
    if max_mag is None:
        max_mag = np.nanpercentile(mag[valid] if valid is not None else mag, 99)
    hsv = np.zeros((*flow.shape[:2], 3), np.uint8)
    hsv[..., 0] = ang * 90 / np.pi
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / max_mag * 255, 0, 255)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    if valid is not None:
        rgb[~valid] = 0
    return rgb


def show_images(images, titles=None, crop=None, figsize_per=(3.6, 3.2), suptitle=None, cmap="gray"):
    """Show grayscale images in [0, 1] side by side on one fixed intensity scale.

    Parameters
    ----------
    images : list of (H, W) arrays
    titles : list of strings, same length
    crop : optional (y0, y1, x0, x1) region shown instead of the full image
    figsize_per : (width, height) of each panel in inches

    Returns
    -------
    the matplotlib Figure
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(images), figsize=(figsize_per[0] * len(images), figsize_per[1]), squeeze=False)
    for ax, img, title in zip(axes[0], images, titles or [""] * len(images)):
        if crop is not None:
            img = img[crop[0]:crop[1], crop[2]:crop[3]]
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    return fig


def busiest_crop(events, shape, size=96):
    """Square crop (y0, y1, x0, x1) centred where the event density is highest.

    Used to zoom into the part of a frame where the blur is, without picking it by hand.
    """
    H, W = shape
    counts = np.bincount(events["y"].astype(np.int64) * W + events["x"].astype(np.int64),
                         minlength=H * W).reshape(H, W).astype(np.float32)
    density = cv2.blur(counts, (size // 2, size // 2))
    cy, cx = np.unravel_index(density.argmax(), density.shape)
    y0 = int(np.clip(cy - size // 2, 0, max(H - size, 0)))
    x0 = int(np.clip(cx - size // 2, 0, max(W - size, 0)))
    return y0, min(y0 + size, H), x0, min(x0 + size, W)
