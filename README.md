# event-perception

Notebooks and a small Python package (`evcam`) for a graduate directed study on event camera
perception methods. The work uses an iniVation DAVIS346 event camera recorded with the DV
software, plus public datasets (N-MNIST, DSEC). Reusable code (loading recordings, event
representations, contrast maximization, metrics, plotting) lives in `evcam/`; the notebooks are
the narrative on top of it and run both locally and on Google Colab.

> **Placeholder:** `allusson` appears in the links below and in the first code cell of each
> notebook. Replace it with the GitHub account that hosts this repo.

## Notebooks

| Notebook | What it covers |
|---|---|
| [`01_representations.ipynb`](notebooks/01_representations.ipynb) | Event representations (raw events, event frames, time surfaces, voxel grids, stored-frame overlay) on N-MNIST or our own DAVIS346 `.aedat4` recordings. |
| [`02_cmax_optical_flow_dsec.ipynb`](notebooks/02_cmax_optical_flow_dsec.ipynb) | Contrast maximization optical flow on DSEC (`thun_00_a`), evaluated against DSEC-Flow ground truth. |
| [`03_deblurring_cmax_edi_efnet.ipynb`](notebooks/03_deblurring_cmax_edi_efnet.ipynb) | Motion deblurring with events on REBlur: CMax + Richardson-Lucy, EDI, and pretrained EFNet, compared by PSNR / SSIM. |
| [`04_frequency_estimation.ipynb`](notebooks/04_frequency_estimation.ipynb) | Frequency estimation from events (event-rate spectrum, per-pixel interval map) on synthetic flicker and on a DAVIS346 propeller recording. |
| [`05_star_cmax.ipynb`](notebooks/05_star_cmax.ipynb) | One global velocity by contrast maximization on a rotating star field (Chin et al., CVPRW 2019; Stellarium on a screen), compared with the ground-truth rotation. |
| [`06_event_generation.ipynb`](notebooks/06_event_generation.ipynb) | How brightness change becomes events: ESIM-style simulation from a 240 fps iPhone slow-motion clip of a fidget spinner, compared with the DAVIS346 recording of the same spinner (hand-picked motion start, side-by-side event images, event rate per pixel). |
| [`07_reconstruction.ipynb`](notebooks/07_reconstruction.ipynb) | Reconstruction from events only (direct integration, high-pass filter, E2VID stub) on simulated events and one REBlur sample. Unpolished, moved from the old notebook 06. |
| [`08_spinner_rotation.ipynb`](notebooks/08_spinner_rotation.ipynb) | Rotation speed of a fidget spinner from DAVIS346 events by two methods: rotation CMax and the event-rate spectrum (harmonic product spectrum), compared over the spin-down. |
| [`09_event_synthesis_diffusion.ipynb`](notebooks/09_event_synthesis_diffusion.ipynb) | Event images generated with the pretrained ControlEvents diffusion models (Hu et al., WACV 2026): from text (compared with plain Stable Diffusion), from a 2D skeleton (dataset samples and one's own photo through OpenPose) and from an SMPL normal map. Demo of released weights, no training. Downloads about 16 GB of weights on first run; use a Colab GPU runtime. |

Signed event images use `bwr` with ON − OFF: red = ON (brighter), blue = OFF (darker).

## Repo layout

```
evcam/                  installable package
  io.py                 load_aedat4, crop_time, frames_look_accumulated, DSEC, REBlur and star dataset loading
  representations.py    time surface, signed event image, SCER (EFNet's event representation)
  cmax.py               iwe, contrast, cmax_patch, patchwise_flow, cmax_rotation, cmax_rotation_2d
  deblur.py             EDI (edi_deblur, edi_select_c), cmax_deblur (CMax + Richardson-Lucy), efnet_deblur
  deblur_eval.py        REBlur test-split evaluation loop with caching
  efnet_arch.py         EFNet model, vendored from github.com/AHupuJR/EFNet (Apache 2.0)
  frequency.py          event_rate_spectrum, pixel_frequency_map, synthetic_flicker
  sources.py            Clip bundle and loaders (synthetic video, video files, DAVIS, REBlur), SOURCES registry
  simulate.py           simulate_events (ESIM-style), make_blur_triplet
  reconstruct.py        direct_integration, highpass_filter (Scheerlinck et al. 2018), fit_scale_offset
  metrics.py            EPE / 3PE, PSNR / SSIM
  viz.py                flow_to_rgb, show_images, busiest_crop, show_events (signed event images)
notebooks/              the notebooks listed above
scripts/download_data.py   DSEC (default), REBlur (`reblur`) and star tracking (`stars`)
tests/                  pytest tests for EDI, SCER, the deconvolution baseline, frequency estimation, sources, simulation and reconstruction
docs/                   DV setup tutorial and paper notes (placeholders)
data/                   recordings and datasets (not in git)
weights/                pretrained model weights (not in git)
```

## Adding a source or method

- **Source:** write a function in `evcam/sources.py` that returns a `Clip` (frames in linear
  intensity with `frame_t`, and/or events in `EVENT_DTYPE`) and add it to `SOURCES`.
- **Deblurring method:** write a function that takes a one-frame `Clip` (blurry frame,
  `exposure`, `events`) and returns the deblurred frame.
- **Reconstruction method:** write a function `(events, shape, ..., t_query) -> (Q, H, W)` log
  intensity in `evcam/reconstruct.py`.

## Local setup (macOS)

From the repo root:

```bash
uv venv --python 3.12        # or: python3.12 -m venv .venv
source .venv/bin/activate
uv pip install -e ".[deblur,dev]"   # deblur: torch + einops for notebook 03; dev: pytest. Plain `pip` works too
uv pip install pip jupyterlab        # or use the VS Code notebook editor with the .venv kernel
```

**Tonic install note.** `01_representations.ipynb` uses [Tonic](https://tonic.readthedocs.io/),
which is deliberately not a dependency of `evcam`: its package metadata pins `numpy<2`, while
everything else here needs `numpy>=2`, so a normal install would downgrade numpy. Install it
without its dependencies, then add the few it actually imports:

```bash
uv pip install tonic --no-deps
uv pip install expelliarmus pbr importRosbag
```

pip will warn that tonic's requirements are not satisfied; that is expected. (The notebook runs
these same two lines itself, so on Colab there is nothing extra to do.)

Then start Jupyter from the repo root with the venv active (`jupyter lab`) and open a notebook
from `notebooks/`. The first code cell of each notebook moves the working directory to the repo
root, so paths like `data/digit4.aedat4` resolve the same way locally and on Colab.

## Running on Colab

Open a notebook directly from GitHub:

```
https://colab.research.google.com/github/allusson/event-perception/blob/main/notebooks/01_representations.ipynb
https://colab.research.google.com/github/allusson/event-perception/blob/main/notebooks/02_cmax_optical_flow_dsec.ipynb
https://colab.research.google.com/github/allusson/event-perception/blob/main/notebooks/03_deblurring_cmax_edi_efnet.ipynb
https://colab.research.google.com/github/allusson/event-perception/blob/main/notebooks/04_frequency_estimation.ipynb
https://colab.research.google.com/github/allusson/event-perception/blob/main/notebooks/05_star_cmax.ipynb
https://colab.research.google.com/github/allusson/event-perception/blob/main/notebooks/06_event_generation.ipynb
https://colab.research.google.com/github/allusson/event-perception/blob/main/notebooks/07_reconstruction.ipynb
https://colab.research.google.com/github/allusson/event-perception/blob/main/notebooks/08_spinner_rotation.ipynb
https://colab.research.google.com/github/allusson/event-perception/blob/main/notebooks/09_event_synthesis_diffusion.ipynb
```

The first code cell clones the repo into the Colab session, changes into it, and runs
`pip install -e .`.

## Data

Everything lives in `data/` at the repo root, which is gitignored.

- **DSEC (`thun_00_a`)**, used by notebook 02: about 300 MB of left-camera events plus about
  12 MB of forward flow ground truth. Fetch it with

  ```bash
  python scripts/download_data.py   DSEC (default) and REBlur (`reblur`)
tests/                  pytest tests for EDI, SCER and the deconvolution baseline
  ```

  which unpacks into `data/dsec/thun_00_a/` and skips anything already downloaded. Notebook 02
  runs this script itself, so this step is only needed if you want the data ahead of time.
- **REBlur**, used by notebook 03: both releases from the EFNet authors (SCER voxels and raw
  events), about 1.2 GB to download and 8 GB unpacked into `data/reblur/`:

  ```bash
  python scripts/download_data.py reblur
  ```

  Notebook 03 also needs the pretrained checkpoint at `weights/EFNet-REBlur.pth`. It is hosted on
  Google Drive and has to be downloaded by hand: use the "pretrained model" link under REBlur in
  the [EFNet README](https://github.com/AHupuJR/EFNet). `weights/` is gitignored.
- **Fidget spinner recording for notebook 06**: one session filmed at the same time by an iPhone
  (240 fps slow motion) and the DAVIS346, side by side. The phone clip goes in
  `data/video_data/spinner.mov` (or `.mp4`, then set `VIDEO_PATH`), the DAVIS clip in
  `data/davis_data/spinner.aedat4`. Notebook 03's own-data cell also reads the DAVIS clip. If
  OpenCV cannot decode an iPhone HEVC file, convert it first with
  `ffmpeg -i input.MOV -c:v libx264 -crf 18 output.mp4`.
- **Star tracking** (Chin et al., CVPRW 2019), used by notebook 05: the original archive (about
  430 MB) from its Internet Archive copy, since the project page is gone; one sequence is unpacked
  into `data/stars/`.
  ```bash
  python scripts/download_data.py stars
  ```
- **ControlEvents**, used by notebook 09: the authors' skeleton and normal-map data (about 0.7 GB
  of zips from the Hugging Face dataset `YixuanHu/ControlEvents_data`), unpacked by the notebook into
  `data/controlevents/`. Each condition (13 joints as text, or a normal map) is paired with an event
  image the authors generated, not a recording. The weights (about 16 GB: the ControlEvents UNets and
  ControlNets plus VAE, text encoder and UNet from Stable Diffusion 1.4 / 1.5) go to
  `weights/controlevents/`. ControlEvents weights and data are MIT; Stable Diffusion weights are
  CreativeML OpenRAIL-M. Locally the notebook also needs
  `pip install diffusers transformers accelerate safetensors controlnet_aux huggingface_hub`.
  For the own-pose cell, put a photo at `data/controlevents/own_pose.jpg`.
- **N-MNIST**, optional in notebook 01: downloaded automatically by Tonic into `data/`.
- **Our own DAVIS346 recordings** (`.aedat4`), used by notebook 01: copy them into `data/`, e.g.
  `data/digit4.aedat4`. Notebook 04 expects `data/propeller.aedat4` and skips its real-data
  section when the file is missing.

**`digit4.aedat4` has no camera frames.** Its frame stream is DV Accumulator output (images
rendered from the events), because the camera's `frames` output was not wired to the file output
when it was recorded. `load_aedat4` warns about this. It is fine for the event representations in
notebook 01, but frame-based deblurring (EDI, EFNet) cannot run on it; notebook 03's own-data cell
stays a placeholder until a clip with APS frames is recorded.

**Recordings are kept out of git.** `.aedat4` files are tens of MB each, so they are not
committed and a fresh clone (including the one Colab makes) does not contain them. On Colab,
either upload the recording into `event-perception/data/` from the Files panel, or keep
recordings in Google Drive and use the commented-out Drive mount in notebook 01's data-source
cell.
