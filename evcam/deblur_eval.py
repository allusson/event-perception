"""Evaluation loops for the deblurring comparison on REBlur (notebook 03).

Kept out of the notebook because they are long-running loops with caching, not something
to read. Results are cached as .npz next to the data, so re-running the notebook is fast;
delete the cache file (or pass `cache=None`) to recompute.
"""

import os
import warnings

import numpy as np

from evcam.deblur import (_divisor, cmax_deblur, edi_oracle_c, edi_select_c, efnet_deblur,
                          event_integral)
from evcam.io import list_reblur, load_reblur_sample
from evcam.metrics import psnr, ssim
from evcam.representations import scer, scer_mask

METHODS = ["blurry", "cmax_rl", "edi_selected", "edi_fixed", "edi_oracle", "efnet", "efnet_own_scer"]


def fit_fixed_c(split="train", root="data/reblur", step=2, cs=np.linspace(0.0, 0.4, 17), cache="auto"):
    """The single c that gives the best mean EDI PSNR on a REBlur split (use the train split).

    A baseline for c selection: one global threshold, fitted with ground truth on training
    data, then applied unchanged to test data.

    Parameters
    ----------
    split, root : as in `list_reblur`
    step : use every `step`-th sample
    cs : candidate c values
    cache : .npz path, "auto" for <root>/fixed_c_<split>.npz, or None

    Returns
    -------
    c : best single c
    cs, mean_psnr : the candidates and the mean PSNR of each (dB)
    """
    cache = os.path.join(root, f"fixed_c_{split}.npz") if cache == "auto" else cache
    if cache and os.path.exists(cache):
        z = np.load(cache)
        if z["step"] == step and np.array_equal(z["cs"], cs):
            return float(z["c"]), z["cs"], z["mean_psnr"]
    total, samples = np.zeros(len(cs)), list_reblur(split, root)[::step]
    for seq, i in samples:
        d = load_reblur_sample(seq, i, split, root)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            E, _ = event_integral(d["events"], d["t_begin"], d["t_end"], (d["t_begin"] + d["t_end"]) / 2,
                                  d["blurry"].shape)
        total += [psnr(np.clip(d["blurry"] / _divisor(E, c), 0, 1), d["sharp"]) for c in cs]
    mean_psnr = total / len(samples)
    c = float(cs[mean_psnr.argmax()])
    if cache:
        np.savez(cache, c=c, cs=cs, mean_psnr=mean_psnr, step=step)
    return c, cs, mean_psnr


def evaluate_reblur(model, device, fixed_c, split="test", root="data/reblur", cmax_every=8,
                    cache="auto", progress=True):
    """PSNR / SSIM of every deblurring method on a REBlur split.

    Methods (keys of the returned dicts, see METHODS):
        blurry          the input, as the reference point
        cmax_rl         `cmax_deblur`; only on every `cmax_every`-th sample (it is slow), NaN elsewhere
        edi_selected    EDI at the c chosen by `edi_select_c` (no ground truth used)
        edi_fixed       EDI at `fixed_c` (one global c fitted on the train split)
        edi_oracle      EDI at the PSNR-maximizing c: uses the ground truth, upper bound only
        efnet           EFNet with the SCER and mask stored in REBlur
        efnet_own_scer  EFNet with SCER and mask computed by evcam from the raw events

    Parameters
    ----------
    model, device : EFNet from `load_efnet` and its torch device
    fixed_c : c for the "edi_fixed" row, from `fit_fixed_c("train")`
    split, root : as in `list_reblur`
    cmax_every : CMax + RL runs on samples 0, cmax_every, 2 * cmax_every, ...
    cache : .npz path, "auto" for <root>/eval_<split>.npz, or None
    progress : print a line every 100 samples

    Returns
    -------
    dict with
        "samples" : list of (sequence, index)
        "psnr", "ssim" : dicts method -> (N,) array, NaN where the method was not run
        "c_selected", "c_oracle" : (N,) arrays
        "n_events" : (N,) events in each exposure window
    """
    cache = os.path.join(root, f"eval_{split}.npz") if cache == "auto" else cache
    samples = list_reblur(split, root)
    if cache and os.path.exists(cache):
        z = np.load(cache)
        if int(z["cmax_every"]) == cmax_every and float(z["fixed_c"]) == fixed_c and len(z["n_events"]) == len(samples):
            return {"samples": samples, "c_selected": z["c_selected"], "c_oracle": z["c_oracle"],
                    "n_events": z["n_events"],
                    "psnr": {m: z[f"psnr_{m}"] for m in METHODS}, "ssim": {m: z[f"ssim_{m}"] for m in METHODS}}

    N = len(samples)
    P = {m: np.full(N, np.nan) for m in METHODS}
    S = {m: np.full(N, np.nan) for m in METHODS}
    c_sel, c_orc, n_ev = np.zeros(N), np.zeros(N), np.zeros(N, dtype=int)
    for k, (seq, i) in enumerate(samples):
        d = load_reblur_sample(seq, i, split, root)
        b, gt, ev, tb, te = d["blurry"], d["sharp"], d["events"], d["t_begin"], d["t_end"]
        n_ev[k] = len(ev)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")           # "no events in window" on the few empty samples
            c_sel[k], _ = edi_select_c(b, ev, tb, te)
            c_orc[k], _ = edi_oracle_c(b, gt, ev, tb, te)
            E, _ = event_integral(ev, tb, te, (tb + te) / 2, b.shape)
        own = scer(ev, tb, te, b.shape)
        out = {"blurry": b,
               "edi_selected": np.clip(b / _divisor(E, c_sel[k]), 0, 1),
               "edi_fixed": np.clip(b / _divisor(E, fixed_c), 0, 1),
               "edi_oracle": np.clip(b / _divisor(E, c_orc[k]), 0, 1),
               "efnet": efnet_deblur(model, b, d["scer"], d["mask"], device),
               "efnet_own_scer": efnet_deblur(model, b, own, scer_mask(own), device)}
        if k % cmax_every == 0:
            out["cmax_rl"] = cmax_deblur(b, ev, tb, te)
        for m, img in out.items():
            P[m][k], S[m][k] = psnr(img, gt), ssim(img, gt)
        if progress and (k + 1) % 100 == 0:
            print(f"  {k + 1}/{N}")

    if cache:
        np.savez(cache, cmax_every=cmax_every, fixed_c=fixed_c, c_selected=c_sel, c_oracle=c_orc, n_events=n_ev,
                 **{f"psnr_{m}": P[m] for m in METHODS}, **{f"ssim_{m}": S[m] for m in METHODS})
    return {"samples": samples, "psnr": P, "ssim": S, "c_selected": c_sel, "c_oracle": c_orc, "n_events": n_ev}
