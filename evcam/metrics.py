"""Evaluation metrics: optical flow now, image quality (for deblurring) later."""

import numpy as np


def endpoint_error(est, gt):
    """Per-pixel endpoint error between two flow fields.

    Parameters
    ----------
    est, gt : (H, W, 2) flow arrays in pixels

    Returns
    -------
    (H, W) float array in pixels (NaN wherever `est` or `gt` is NaN)
    """
    return np.linalg.norm(est - gt, axis=-1)


def epe(est, gt, mask):
    """Mean endpoint error in pixels over the pixels where `mask` is True.

    Parameters
    ----------
    est, gt : (H, W, 2) flow arrays in pixels
    mask : (H, W) bool, pixels to evaluate (e.g. GT valid and estimate exists)
    """
    return endpoint_error(est, gt)[mask].mean()


def n_pixel_error(est, gt, mask, n=3):
    """Fraction (0 to 1) of masked pixels whose endpoint error is > n pixels.

    n=3 gives the DSEC-Flow "3PE" metric.
    """
    return (endpoint_error(est, gt)[mask] > n).mean()


def psnr(img, ref, data_range=1.0):
    """Peak signal-to-noise ratio in dB between two (H, W) images. Stub for the deblurring work."""
    raise NotImplementedError("psnr will be added with the deblurring notebooks")


def ssim(img, ref, data_range=1.0):
    """Structural similarity between two (H, W) images. Stub for the deblurring work."""
    raise NotImplementedError("ssim will be added with the deblurring notebooks")
