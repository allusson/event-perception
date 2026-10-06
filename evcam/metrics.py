"""Evaluation metrics: optical flow (EPE, N-pixel error) and image quality (PSNR, SSIM)."""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


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
    """Peak signal-to-noise ratio in dB between two images of the same shape.

    Parameters
    ----------
    img, ref : (H, W) float arrays (estimate and ground truth)
    data_range : value range of the images (1.0 for images in [0, 1])
    """
    return float(peak_signal_noise_ratio(ref, img, data_range=data_range))


def ssim(img, ref, data_range=1.0):
    """Structural similarity (Wang et al. 2004) between two (H, W) images, in [-1, 1].

    Uses the standard settings of the original paper (Gaussian window, sigma 1.5), which
    are what deblurring papers report.
    """
    return float(structural_similarity(ref, img, data_range=data_range, gaussian_weights=True,
                                       sigma=1.5, use_sample_covariance=False))
