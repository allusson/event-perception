# Removed notebook notes

Markdown text taken out of the notebooks in `notebooks/` when they were thinned to a skeleton,
copied verbatim and grouped by notebook and section. Where a section was shortened and not
emptied, its whole original text is kept here.

## 01_representations.ipynb

### Introduction (under the notebook title)

This notebook is a first hands-on pass at event camera data, following the reading of
Gallego et al., *"Event-based Vision: A Survey"* (IEEE TPAMI 2020). It uses the
[Tonic](https://tonic.readthedocs.io/) library for the standard representations, and can run on
either a public dataset or recordings from our own DAVIS346 camera.

Goals of this notebook:
1. Load event data (N-MNIST, or our own `.aedat4` recordings) and inspect the raw event stream format.
2. Reproduce a few of the standard event *representations* described in the survey
   (Section 3.1): raw events in space-time, event frames / 2D histograms, and time surfaces.
3. Look at voxel grids and at pairing events with the frame stream stored in our own
   recordings (for `digit4` that stream is DV Accumulator output, not camera frames; see section 7).

**Data sources.** N-MNIST downloads automatically, is small (34x34 pixels), and needs no
registration, which makes it a good starting point. Our own recordings come from an iniVation
DAVIS346 (346x260 pixels) via the DV software, saved as `.aedat4` files. Switch between them in
the *Choose a data source* cell below; every cell after that works for either.

### 2. Choose a data source

Set `DATA_SOURCE` to `"nmnist"` for the public dataset, or `"davis"` for one of our own
recordings. For our recordings:

- Put the `.aedat4` file in the repo's `data/` folder and set `RECORDING_PATH`. Recordings are
  kept out of git, so on Colab either upload the file into `event-perception/data/` (Files panel
  on the left) or put it in Google Drive, mount Drive (commented-out lines below), and point
  `RECORDING_PATH` at it.
- A raw recording is several seconds long and contains hundreds of thousands of events per
  second, far more than an N-MNIST sample. The representations below are meant to show a short
  slice of time, so `T_START_S` and `WINDOW_S` pick which slice to analyze. The event-rate plot
  in the next section helps choose `T_START_S`.
- To show a different digit, only this cell needs to change.

#### Loading `.aedat4` recordings

DV saves each stream that is wired to its file output inside the `.aedat4` file: the events, a
frame stream, and (if connected) the IMU. The `aedat` package decodes these into packets.
`load_aedat4` (in `evcam/io.py`) collects the events into the same structured array layout Tonic
uses for N-MNIST, `(x, y, t, p)` with `p` as 0/1, so all the cells after this point work unchanged
on either data source. Timestamps in the file are absolute (microseconds since the Unix epoch),
so they are shifted to start at 0.

Each frame is returned as a dict with its `image`, its timestamp `t`, and the start and
end of its exposure (`exposure_begin_t`, `exposure_end_t`), all on the same zeroed clock as the
events. For real APS frames the exposure window is what a deblurring step (EDI) needs in order to
pick out the events that fired while the frame was being exposed.

**The frames in `digit4.aedat4` are not APS frames.** The camera's own `frames` output was not
wired to the file output when this clip was recorded; the Accumulator module was, so the stored
frames are images DV rendered from the events. `load_aedat4` warns about this (the check is
`frames_look_accumulated` in `evcam/io.py`), and section 7 comes back to what it means. `crop_time` (same file) cuts out a short
time window of events and re-zeroes its timestamps.

### 3. Load a sample and inspect the raw event stream

An event camera does not output frames. Each event is a tuple `(x, y, t, p)`:
- `x, y`: pixel location
- `t`: timestamp (microseconds)
- `p`: polarity, whether the log-brightness increased (ON) or decreased (OFF)

This is exactly the `e_k = (x_k, t_k, p_k)` notation used in Section 2.4 of the survey.

*Cell without a heading (cell 9), in "3. Load a sample and inspect the raw event stream":*

Notice how sparse and asynchronous this is. A 34x34 N-MNIST sample of a digit produces a few
thousand events spread over a fraction of a second, rather than a fixed grid of pixel
intensities at a fixed frame rate. A real 346x260 recording produces far more, since there are
~80x more pixels and every edge in the scene (not just the digit) generates events as it moves.

### 4. Representation 1 — raw events in space-time

The most direct way to look at events is as points in an (x, y, t) volume, colored by
polarity. This is Fig. 3(a) in the survey.

### 5. Representation 2 — event frames (2D histograms)

The most common way to make events compatible with standard CV tooling (CNNs, etc.) is to
accumulate events over a short time window into a 2D image. Here each frame is
`ON count - OFF count` per pixel, using Tonic's `ToFrame` transform to bin the stream into
a handful of time windows.

### 6. Representation 3 — time surface

A time surface stores, per pixel, a value that decays with how long ago that pixel last
fired (Section 3.1 of the survey). It captures short-term motion history in a single 2D
image instead of a stack of frames.

The implementation (`time_surface` in `evcam/representations.py`) is vectorized so it stays fast on real recordings with tens of
thousands of events. Because the decay value only grows with time, "the most recent event at
each pixel" is the same as "the largest decay value at each pixel", which `np.maximum.at`
computes in one call instead of a Python loop.

`tau` sets how long the trail is. For N-MNIST's short samples ~5 ms works well; for a 50 ms
window of real data a longer `tau` (~15 ms) shows more of the motion history.

### 7. Voxel grids and multi-modal (event + frame) data

- **Voxel grids**: instead of collapsing time into a handful of 2D frames, a voxel grid
  keeps a 3D (x, y, t-bin) histogram, which several of the deep-learning optical flow and
  SLAM methods in the paper list use as network input (e.g. EV-FlowNet-style pipelines).
  Tonic provides `transforms.ToVoxelGrid` as a drop-in replacement for `ToFrame` above.
- **Multi-modal event + frame data**: DAVIS cameras output events and low-rate grayscale
  (APS) frames from the same pixel array, which is the pairing most of the SLAM/VIO papers in the
  list rely on. When running on our own recordings, the last cell of this section shows the stored
  frame closest to the analysis window next to the events. For `digit4` that stored frame is DV
  Accumulator output rather than an APS frame, so the comparison there is events against an
  image made from events; the cell labels it accordingly.

#### Why there is no EDI demo on this recording

EDI (Event-based Double Integral, Pan et al. 2019) sharpens a motion-blurred frame using the
events that fired during its exposure. It rests on one physical fact: the frame is the time
integral of the light that reached each pixel while the shutter was open, and the events record
how that light changed over the same interval. Dividing one by the other recovers a sharp image.

The frames in `digit4.aedat4` do not satisfy that. They are DV Accumulator output: each image is
built by adding a fixed step per event to a flat background, with decay. The evidence is in the
data itself: the grey levels pile up on multiples of 38 (0.15 of full scale, the Accumulator's
default event contribution), consecutive frames differ at only ~6% of pixels and that difference
is the signed event image of the same 33 ms, and the stored "exposure" equals the frame period
exactly. Nothing in such a frame integrated light, so there is no blur for the events to explain,
and running EDI on it would use the events to "correct" an image made from those same events.
The result would be circular, not a deblurred frame.

The EDI demo therefore lives in `03_deblurring_cmax_edi_efnet.ipynb`, on the REBlur dataset, which
has real blurry APS frames with ground truth. Re-recording a clip with the camera's `frames` output
wired to the file output would make an own-data version possible.

### 8. Summary and next steps

What this notebook covers: loading event data (N-MNIST or our own DAVIS346 recordings),
understanding the `(x, y, t, p)` event format, and reproducing three of the standard
representations from the survey (raw events, event frames, time surfaces), plus voxel grids
and a first look at pairing events with the frame stream stored in the recording (Accumulator
output for `digit4`, see the note at the end of section 7).

What it does not cover yet: an actual optical flow / SLAM / generative algorithm running on
this data. The papers in the reference list (E-RAFT, EV-FlowNet, "Secrets of Event-Based Optical
Flow" for optical flow; ESVO2 / DEIO for SLAM/VO; ControlEvents for generative modeling)
each have public code that would be the next thing to get running once one task is
selected. The `load_aedat4` / `crop_time` functions in `evcam/io.py` are reusable for that work.

## 02_cmax_optical_flow_dsec.ipynb

### Introduction (under the notebook title)

Model-based optical flow via contrast maximization (Gallego, Rebecq, Scaramuzza, CVPR 2018), run on one DSEC training sequence and evaluated against DSEC-Flow ground truth.

Setup used here: one 100 ms window, constant flow per patch, variance as the contrast objective, coarse-to-fine grid search.

### 1. Setup and data download

`hdf5plugin` (installed as an `evcam` dependency and imported by `evcam/io.py`) is needed because DSEC event files are Blosc/ZSTD compressed; h5py can't read them without it. The download (`scripts/download_data.py`, which skips files already in `data/dsec/thun_00_a/`) pulls only the left event camera (~300 MB) plus forward flow ground truth (~12 MB) for thun_00_a, which is one of the smaller sequences with flow GT.

### 2. Loading events and ground truth

The loading functions used here (`open_dsec_events`, `load_rectify_map`, `load_flow`, `get_events`, ...) are in `evcam/io.py`.

Relevant DSEC format details:
- `events.h5` holds `x, y, t, p` (t in µs), a `t_offset` to convert to the global clock used by the flow timestamps, and `ms_to_idx`, a millisecond-to-index lookup so we can slice a time window without reading all ~134M events.
- Events are stored in raw (distorted) sensor coordinates, but GT flow is in the rectified left-camera frame, so events are mapped through `rectify_map.h5` first.
- GT flow is a 16-bit PNG per 100 ms window: R = x flow, G = y flow, B = valid mask, decoded as `(value - 2^15) / 128` pixels. It's the displacement from `t_k` to `t_k+1`, and it's sparse (LiDAR-based, static scene only).
- OpenCV loads channels as BGR, hence the `[..., ::-1]`. The sorted PNG files line up row by row with the timestamp file.

### 3. One 100 ms window

Pick GT window `K` and pull the matching events. Time is normalized to [0, 1) so a flow `d` (in pixels per window) is directly comparable to the GT displacement.

Left: GT flow with the standard HSV encoding (hue = direction, brightness = magnitude). Right: where GT exists. The unwarped event image is shown in section 4 once the accumulation function is introduced.

### 4. Image of Warped Events (IWE) and the contrast objective

Each event is moved back to the window start under a candidate flow `d`: `x' = x - d * t_norm`. Warped events are accumulated with bilinear voting (each event splits its weight over the 4 nearest pixels) and blurred slightly. Both make the objective smoother in `d`, which matters for any search or gradient-based optimizer. Polarity is ignored, which is the standard choice for flow.

Contrast is the variance of the IWE. Correct `d` stacks events from the same edge onto the same pixels, so variance goes up.

Both functions, `iwe` and `contrast`, are in `evcam/cmax.py`. `origin` and `shape` let the IWE be computed on a local canvas around a patch instead of the full frame.

### 5. Single patch: full contrast landscape

Exhaustive grid search over `d` for one 64×64 patch to see the shape of the objective. The canvas is padded so events warped past the patch edge aren't dropped. The patch location was chosen because it has both textured events and decent GT coverage.

The heatmap shows how peaked (or not) the objective is; an elongated peak means flow is poorly constrained along that axis (aperture problem, e.g. a patch dominated by edges of one orientation). The IWE panels are a zoomed crop of just this patch (dashed box = patch, the dark border is the padding), blurred with `sigma=1`, which is why they look different from the full-frame image. The leftmost panel shows where the patch sits. Comparing d = 0 to the best d shows the blurred vs. sharpened edges. The GT median is only a rough reference since true flow varies within the patch (forward driving gives a diverging field).

### 6. Patch-wise flow over the full frame

Same objective, applied independently to a grid of 64×64 patches (`patchwise_flow` in `evcam/cmax.py`, which calls `cmax_patch` on each patch). Full grid search per patch is too slow, so this uses a coarse-to-fine search: step 4 px over ±40, then step 1 px around the best coarse point. Patches with too few events are skipped (left as NaN).

Patch size is the main tradeoff. Larger patches have more events and a better-conditioned objective, but the constant-flow assumption breaks down more.

### 6b. Before vs. after motion compensation (full frame)

Each event is warped with the flow of the patch it came from (events in skipped patches are left unwarped), then accumulated into one full-frame IWE. Left is the d = 0 image from section 4, right is the compensated one, both with no blur and the same display scaling. If the per-patch flows are right, edges should collapse into thin lines. Tile seams and residual blur show where constant flow per patch doesn't hold.

### 7. Evaluation against GT

DSEC-Flow metrics on pixels where GT is valid and a patch estimate exists: EPE (mean endpoint error, px) and 3PE (% of pixels with error > 3 px). Zero flow is included as a sanity baseline. The metric functions are in `evcam/metrics.py`.

Note this isn't directly comparable to the benchmark leaderboard, which uses the test set and dense per-pixel predictions.

*Cell without a heading (cell 18), in "7. Evaluation against GT":*

Arrows over the no-warp event image, one per patch: GT patch median (thick cyan) and CMax (thin red, drawn on top). Scaled by `ARROW_SCALE` for visibility. Makes direction errors easier to spot than the color maps.

### 8. Notes and next steps

- Constant flow per patch is a big limitation. Forward driving creates a diverging field, so the assumption is worst in patches with large depth changes and near the image edges, where flow is largest.
- Low-texture patches (e.g. road with only lane markings) give weakly constrained objectives.
- Current knobs: patch size, window length (fixed at 100 ms here to match GT), blur `sigma`, and search range.
- Possible extensions: other objectives (e.g. sum of squares, gradient magnitude), gradient-based optimization instead of grid search, richer motion models (affine, rotation), multi-scale patches, and the Shiba et al. 2022 improvements on this same dataset.
- For comparison: the local plane-fitting method (Benosman et al. 2014) on the same window.

## 03_deblurring_cmax_edi_efnet.ipynb

### Introduction (under the notebook title)

A frame camera integrates light over its exposure, so anything that moves during the exposure is
smeared. An event camera watching the same scene records *how* each pixel's brightness changed
during that exposure, with microsecond timing. This notebook deblurs frames with three methods that
use those events, on the REBlur dataset (real blurry DAVIS frames with ground truth).

**A ladder of assumptions.** The three methods use the same physics in different roles, and each
step down the ladder assumes less and needs more data:

| | What it assumes | What it needs | Where the event physics sits |
|---|---|---|---|
| **CMax + deconvolution** | A motion model: every patch moves with one constant velocity, and blur is a convolution with a line | nothing | Events are used only to *estimate motion* (they line up when warped along the right flow). The deblurring itself never sees them. |
| **EDI** (Pan et al., CVPR 2019) | No motion model. One contrast threshold `c` for the whole frame: each event is a log-intensity step of size `c` | one number, `c` | The event generation model *is* the deblurring formula: blurry = sharp × mean over the exposure of exp(c · event integral). |
| **EFNet** (Sun et al., ECCV 2022) | Neither. A network learns the mapping | paired blurry / sharp training data | Only in the *input representation*: SCER feeds the network the same event integrals EDI uses, and the network learns what to do with them, including where the events are noisy. |

Sections 3 to 5 run each method on a few samples, section 6 compares them over the REBlur test
split, and section 7 is a placeholder for our own DAVIS346 data.

*Cell without a heading (cell 3), in "1. Setup and data":*

Everything reusable is in `evcam`: `deblur.py` (the three methods), `efnet_arch.py` (the EFNet model,
vendored from the authors' repo), `representations.py` (SCER), `metrics.py` (PSNR, SSIM),
`deblur_eval.py` (the test-split loop) and `io.py` (REBlur loading).

Requirements beyond the base install: `pip install -e ".[deblur]"` (torch, einops), the REBlur data
(about 1.2 GB download, 8 GB unpacked; fetched by the cell below and skipped if already there), and
the pretrained checkpoint `weights/EFNet-REBlur.pth`, which has to be downloaded by hand from the
Google Drive link in the [EFNet README](https://github.com/AHupuJR/EFNet) (on Colab, upload it into
`event-perception/weights/`).

### 2. REBlur samples

REBlur was recorded with a DAVIS camera in a light-controlled lab: 260 × 320 grayscale frames with
a 46 ms exposure, and a sharp ground-truth frame for each. There are two releases of the same
sequences and `load_reblur_sample` reads both: one with the events already converted to SCER (what
EFNet consumes) and one with the raw events and each frame's exposure start and end, which EDI and
CMax need.

Three test samples are used throughout, one per kind of motion in the dataset: an object moving
on a line, an object moving on a circle, and camera motion. The yellow box is the zoom region used
in later figures, placed automatically where the event density is highest.

*Cell without a heading (cell 7), in "2. REBlur samples":*

**SCER** (Symmetric Cumulative Event Representation) is EFNet's event input: 6 channels, each the
signed event count between the exposure midpoint and one of 6 time points, 3 toward the start and
3 toward the end. `evcam.representations.scer` rebuilds it from raw events. It has to match what
EFNet was trained on, because section 7 computes it for our own recordings, so it is checked here
against the voxels stored in REBlur. The small remaining difference is a boundary convention:
EFNet places the 6 slices between the first and last *event* in the exposure, this implementation
between the exposure start and end, so a few events on slice boundaries land one slice over.
Replicating EFNet's convention reproduces the stored voxels exactly.

### 3. CMax + Richardson-Lucy deconvolution

The classical recipe: estimate the motion, turn it into a blur kernel, deconvolve.

1. **Motion from events.** `patchwise_flow` from notebook 02 runs contrast maximization on the
   events inside the exposure. With time normalized over the exposure, the flow it returns is in
   pixels per exposure, which is the blur vector.
2. **Kernel.** Each 64 px patch gets a linear motion PSF of that length and direction.
3. **Deconvolve and blend.** Richardson-Lucy (10 iterations) on each tile; four half-shifted patch
   grids with Hann windows give overlapping tiles that blend without seams.
4. **Event gate.** The deconvolved result is kept only near pixels that fired events. A pixel
   without events did not change during the exposure, so it is not blurred.

Left: the estimated blur vectors (one arrow per patch with enough events). Middle: the result.

*Cell without a heading (cell 11), in "3. CMax + Richardson-Lucy deconvolution":*

The flow field is the informative part. Where a patch is filled by one moving surface the arrows
agree with the visible smear; patches with few events or a single edge orientation give arrows of
the wrong length or direction, and the deconvolution then adds ringing instead of removing blur.
The iteration count and the event gate were chosen on the REBlur train split. Without the gate this
method scores *below* the blurry input, because in most REBlur scenes a patch holds a moving object
in front of a static background and the object's PSF gets applied to both.

### 4. EDI: Event-based Double Integral

An event fires when a pixel's log intensity changes by the contrast threshold `c`. So the sharp
image at time `t` is the sharp image at a reference time, times `exp(c · E(t))`, where `E(t)` is the
signed event count between the two times. Averaging over the exposure gives the blurry frame:

$$B = L(t_\text{ref}) \cdot \frac{1}{T}\int_{t_0}^{t_0+T} \exp\big(c\,E(t)\big)\,dt
\quad\Longrightarrow\quad L(t_\text{ref}) = B \,/\, \text{divisor}$$

No motion model and no kernel: every pixel is corrected by its own events. The one free parameter
is `c`, and two ways of setting it are shown:

- **Selected c** (`edi_select_c`): no ground truth. Minimizes total variation of the result minus a
  weighted correlation between the result's edges and an edge map made from the events, following
  the EDI paper's criterion. The weight was tuned on the REBlur *train* split only.
- **Oracle c** (`edi_oracle_c`): the `c` that maximizes PSNR against the ground truth. This is not a
  method, it is the ceiling for any way of choosing a single `c` per frame.

*Cell without a heading (cell 14), in "4. EDI: Event-based Double Integral":*

The curves below show why the two disagree. Blue is the selection objective (its minimum is the
selected `c`), red is PSNR against ground truth (its maximum is the oracle `c`).

*Cell without a heading (cell 16), in "4. EDI: Event-based Double Integral":*

**The gap between selected and oracle c.** On these samples, and on the test split in section 6, the
GT-free criterion almost always returns the smallest `c` it is allowed to (0.05), while the
PSNR-optimal `c` is spread between 0 and about 0.3. The reason is visible in the blue curves: they
have no interior minimum near the red peak. The edge-correlation term keeps improving as `c` grows,
because every change EDI makes sits on event pixels, and the total-variation term grows smoothly
with `c` instead of turning up sharply once the result overshoots. Their weighted sum therefore
either stays at the lower bound or runs to a large `c`, depending on the weight, and the weight that
is best on the train split is the one that keeps it at the lower bound. So in practice the selected
`c` is a cautious constant: it under-corrects, which is safe but leaves blur.

The sweep shows the other side: beyond the best `c` the edges overshoot and event noise becomes
visible as speckle, because each noise event is now treated as a real brightness step.

### 5. EFNet: pretrained inference

EFNet is a two-stage U-Net with an event branch. It takes the blurry frame, the SCER of the events
in the exposure and a binary mask of the pixels that fired, and fuses image and event features with
cross-modal attention. The checkpoint is the authors' model fine-tuned on the REBlur train split.
`efnet_deblur` follows the preprocessing of the EFNet repo: frame in [0, 1] replicated to 3
channels, SCER divided by its max abs value, mask in {0, 1}, second (final) stage output.

### 6. Comparison on the REBlur test split

PSNR and SSIM against ground truth for every method over all 903 test samples
(`evaluate_reblur` in `evcam/deblur_eval.py`). Notes on what is and is not evaluated:

- **CMax + RL is evaluated on a fixed subset**: every 8th test sample (113 frames), because it
  takes about 5 s per frame. The table has a second pair of columns with *all* methods on that same
  subset, so the CMax row can be compared like for like.
- **EDI, fixed c** is an extra row: one global `c`, the value with the best mean PSNR on the train
  split, applied unchanged to the test split. Like the selected `c`, it uses no test ground truth.
- **EFNet, our SCER + mask** runs the network on SCER computed here from raw events, which is what
  section 7 has to rely on.
- The first run takes about half an hour on a MacBook Air (M-series, MPS); results are cached in
  `data/reblur/eval_test.npz`, so later runs are instant. Delete that file to recompute.

*Cell without a heading (cell 22), in "6. Comparison on the REBlur test split":*

**Reading the table.**

- **EFNet** reproduces the paper: 38.07 dB / 0.975 here against 38.12 / 0.975 published. Feeding it
  SCER computed by `evcam` from raw events gives the same result as the stored SCER, so the
  representation is safe to use on new recordings.
- **EDI** improves on the input by 0.4 dB with the GT-free `c`, 0.7 dB with one constant `c` fitted
  on train, and 1.1 dB with the oracle `c`. The GT-free criterion does worse than a constant, for the
  reason given in section 4.
- **EDI here is about 1.1 dB below the paper's EDI row even with the oracle c** (35.42 against 36.52).
  This is not explained. The published number comes from the EDI authors' implementation, whose
  details (how `c` is chosen, any denoising of events or output, which instant is reconstructed) are
  not specified in the EFNet paper. One thing measured on the train split that points at timing:
  reconstructing slightly after the exposure midpoint (60% of the way through) scores about 0.3 dB
  higher than the midpoint, which suggests the ground-truth frames are not exactly mid-exposure.
  That was not used here; the midpoint is the principled default.
- **CMax + RL** gains about 0.55 dB on its subset, between EDI with selected `c` and EDI with a fixed
  `c`, and only with the event gate.
- All of these gains are averages over a dataset where much of each frame is static, so the input
  already scores 34.3 dB. The box plot below shows the spread per sample.

#### Failure modes, as observed

**CMax + RL**
- A patch with a moving object on a static background gets one PSF for both. Inside the event gate
  the background next to the object is "deblurred" with the object's motion, which shows as ringing
  and ghost edges parallel to the motion.
- Patches with few events, or with edges of one orientation only, produce blur vectors of the wrong
  length or direction; deconvolving with a wrong kernel is worse than doing nothing.
- Curved motion (the circle sequences) is not a straight line within a patch, and the linear PSF
  only removes part of the blur.
- Richardson-Lucy amplifies noise and rings at tile content it cannot explain; more iterations make
  this worse, which is why only 10 are used.

**EDI**
- Everything depends on `c`, and one `c` per frame is itself an approximation: thresholds differ
  between ON and OFF events and from pixel to pixel.
- Too small a `c` leaves blur; too large a `c` overshoots edges and turns noise events into speckle.
  Where events are missing (the sensor's refractory period at fast edges) the correction is too
  weak, leaving a residual smear.
- It can only change pixels that fired events, so it never damages static regions, but it also
  inherits every event artifact directly: there is no regularization at all.
- The GT-free choice of `c` is the weak point here, not the model: the oracle shows the model has
  about 0.7 dB more to give than the selected `c` delivers.

**EFNet**
- By far the best on REBlur, but it was fine-tuned on REBlur's own train split: same camera, same
  lab, same kinds of motion. The number says little about a different sensor, lighting or scene.
- It depends on its inputs matching training: without the event mask its PSNR on this test split
  drops to about 34.8 dB (measured on a quarter of the split), barely above the input. The SCER
  normalization and channel order matter in the same way.
- Its errors are not interpretable. Where the two model-based methods fail there is a reason to
  point at (wrong flow, wrong `c`); here there is only the residual.

### 7. Own data (placeholder)

The plan is to run EDI and EFNet (with our own SCER) on a frame from our DAVIS346, picking a frame
whose exposure lies fully inside the event stream. That is qualitative only, since there is no
ground truth.

**This cell currently skips.** The only recording so far, `digit4.aedat4`, does not contain camera
frames: its frame stream is DV Accumulator output, images rendered from the events, recorded
because the camera's own `frames` output was not wired to the file output (see notebook 01,
section 7). EDI and EFNet need a frame that physically integrated light during its exposure;
running them on an event-built image would be circular. The cell checks this with
`frames_look_accumulated` and runs as is once a recording with real APS frames is available. A
better-lit clip with APS frames is planned.

### 8. Summary

- The three methods form a ladder. CMax + deconvolution assumes a motion model and uses events only
  to find the motion. EDI drops the motion model and uses the event generation model directly, at
  the price of one unknown threshold. EFNet learns the mapping, with the same event integrals as
  its input.
- On REBlur the ranking follows the ladder: about +0.5 dB for CMax + RL, +0.4 to +1.1 dB for EDI
  depending on how `c` is chosen, +3.8 dB for EFNet.
- The model-based methods fail for reasons that can be named: one PSF per patch for CMax, one `c`
  per frame (and no good way to pick it without ground truth) for EDI. EFNet's advantage comes with
  the caveat that it is evaluated on the distribution it was fine-tuned on.
- Open items: the 1.1 dB gap between this EDI implementation and the published EDI number on REBlur;
  a c-selection rule that tracks the oracle; and own-data results, which need a recording with APS
  frames.
