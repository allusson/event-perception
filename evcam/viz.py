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
