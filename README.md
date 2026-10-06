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

## Repo layout

```
evcam/                  installable package
  io.py                 load_aedat4, crop_time, frames_look_accumulated, DSEC and REBlur loading
  representations.py    time surface, signed event image, SCER (EFNet's event representation)
  cmax.py               iwe, contrast, cmax_patch, patchwise_flow
  deblur.py             EDI (edi_deblur, edi_select_c), cmax_deblur (CMax + Richardson-Lucy), efnet_deblur
  deblur_eval.py        REBlur test-split evaluation loop with caching
  efnet_arch.py         EFNet model, vendored from github.com/AHupuJR/EFNet (Apache 2.0)
  metrics.py            EPE / 3PE, PSNR / SSIM
  viz.py                flow_to_rgb, show_images, busiest_crop
notebooks/              the notebooks listed above
scripts/download_data.py   DSEC (default) and REBlur (`reblur`)
tests/                  pytest tests for EDI, SCER and the deconvolution baseline
docs/                   DV setup tutorial and paper notes (placeholders)
data/                   recordings and datasets (not in git)
weights/                pretrained model weights (not in git)
```

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
- **N-MNIST**, optional in notebook 01: downloaded automatically by Tonic into `data/`.
- **Our own DAVIS346 recordings** (`.aedat4`), used by notebook 01: copy them into `data/`, e.g.
  `data/digit4.aedat4`.

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
