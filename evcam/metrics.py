"""Evaluation metrics: optical flow (EPE, N-pixel error), image quality (PSNR, SSIM) and event image
comparison (Chamfer distance, tolerant precision / recall / F1, motion selectivity)."""

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
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


def _dist_to(mask):
    """Distance in pixels from every pixel to the nearest True pixel of `mask` (inf if there is none)."""
    return distance_transform_edt(~mask) if mask.any() else np.full(mask.shape, np.inf)


def chamfer_distance(a, b):
    """Symmetric Chamfer distance in pixels between the True pixels of two (H, W) bool images.

    Mean distance from each pixel of `a` to the nearest pixel of `b`, and from `b` to `a`, averaged
    (Barrow et al. 1977, with Euclidean distance transforms). NaN if either image is empty.
    """
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if not a.any() or not b.any():
        return np.nan
    return float(0.5 * (_dist_to(b)[a].mean() + _dist_to(a)[b].mean()))


def tolerant_prf(pred, ref, tol):
    """Precision, recall and F1 between two (H, W) bool images with a distance tolerance in pixels.

    precision = share of `pred` pixels with a `ref` pixel within `tol`; recall = share of `ref`
    pixels with a `pred` pixel within `tol`; F1 their harmonic mean. As for boundary maps (Martin et
    al. 2004), but without their one-to-one matching. NaN where a share is undefined (empty image).
    """
    pred, ref = np.asarray(pred, bool), np.asarray(ref, bool)
    p = (_dist_to(ref)[pred] <= tol).mean() if pred.any() else np.nan
    r = (_dist_to(pred)[ref] <= tol).mean() if ref.any() else np.nan
    f = np.nan if np.isnan(p) or np.isnan(r) else (0.0 if p + r == 0 else 2 * p * r / (p + r))
    return float(p), float(r), float(f)


def segment_mask(shape, xy, segments, radius):
    """(H, W) bool: pixels within `radius` px of the given segments (pairs of joint indices into
    `xy`, an (N, 2) array of pixel coordinates; a segment with a NaN joint is left out)."""
    m = np.zeros(shape, np.uint8)
    for a, b in segments:
        if np.isfinite(xy[[a, b]]).all():
            pa, pb = (tuple(int(round(v)) for v in xy[j]) for j in (a, b))
            cv2.line(m, pa, pb, 1, max(1, int(round(2 * radius))))
            cv2.circle(m, pa, int(round(radius)), 1, -1)
            cv2.circle(m, pb, int(round(radius)), 1, -1)
    return m.astype(bool)


def moving_segments(xy_seq, k, segments, move_px):
    """Split `segments` into (moving, still) at frame k of `xy_seq` ((T, N, 2) joints in px, NaN =
    missing): a segment moves if one of its joints moved more than `move_px` between frame k and a
    neighbouring frame (k - 1 or k + 1). Segments with a missing joint at k are in neither list."""
    moving, still = [], []
    for seg in segments:
        if not np.isfinite(xy_seq[k, list(seg)]).all():
            continue
        d = [np.linalg.norm(xy_seq[j, list(seg)] - xy_seq[k, list(seg)], axis=1) for j in (k - 1, k + 1)
             if 0 <= j < len(xy_seq)]
        d = np.concatenate(d) if d else np.array([np.nan])
        (moving if np.nanmax(d, initial=-np.inf) > move_px else still).append(seg)
    return moving, still


def motion_selectivity(events, xy, moving, still, radius):
    """Event density near the moving segments over event density near the still ones.

    events : (H, W) bool event pixels; xy : (N, 2) joints in px; moving, still : lists of segments
    (see `moving_segments`); radius : px. Density = event pixels / pixels of the region; the still
    region excludes pixels that are also near a moving segment; at least one event pixel is counted
    there, so the ratio stays finite when the still limbs have no events. > 1: events concentrate on
    what moves. NaN if either region is empty.
    """
    events = np.asarray(events, bool)
    m = segment_mask(events.shape, xy, moving, radius)
    s = segment_mask(events.shape, xy, still, radius) & ~m
    if not m.any() or not s.any():
        return np.nan
    return float(events[m].mean() / (max(events[s].sum(), 1) / s.sum()))
