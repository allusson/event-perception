"""Download the datasets the notebooks use: DSEC (notebook 02) and REBlur (notebook 03).

DSEC (default). Pulls only the left event camera (~300 MB) plus forward flow ground truth (~12 MB) for
one sequence (default thun_00_a) and unpacks them into:

    data/dsec/<sequence>/
        events_left/events.h5
        events_left/rectify_map.h5
        flow/*.png
        <sequence>_optical_flow_forward_timestamps.txt

REBlur (Sun et al., "Event-Based Fusion for Motion Deblurring with Cross-modal Attention",
ECCV 2022). Pulls both releases (~700 MB and ~500 MB) and unpacks them into data/reblur/:

    REBlur.zip            per-sequence h5 files with blurry/sharp images and SCER voxels
    REBlur_rawevents.zip  the same sequences with the raw events

Files that already exist are skipped, so it is safe to run again.

Usage (from anywhere):
    python scripts/download_data.py
    python scripts/download_data.py --sequence thun_00_a --data-dir data
    python scripts/download_data.py reblur
"""

import argparse
import os
import subprocess
import zipfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_URL = "https://download.ifi.uzh.ch/rpg/DSEC/train"
REBLUR_URL = "https://data.vision.ee.ethz.ch/csakarid/shared/EFNet"
REBLUR_FILES = ["REBlur.zip", "REBlur_rawevents.zip"]


def download(url, dest):
    if os.path.exists(dest):
        print(f"  have     {os.path.basename(dest)}")
        return
    print(f"  download {os.path.basename(dest)}")
    tmp = dest + ".part"  # so an interrupted download is never mistaken for a finished one
    subprocess.run(["curl", "-fL", "-C", "-", "--progress-bar", "-o", tmp, url], check=True)  # -C -: resume a partial file
    os.replace(tmp, dest)


def download_dsec(sequence, data_dir):
    out = os.path.join(data_dir, "dsec", sequence)
    os.makedirs(out, exist_ok=True)
    print(f"DSEC {sequence} -> {os.path.relpath(out)}")

    files = [f"{sequence}_events_left.zip",
             f"{sequence}_optical_flow_forward_event.zip",
             f"{sequence}_optical_flow_forward_timestamps.txt"]
    for f in files:
        download(f"{BASE_URL}/{sequence}/{f}", os.path.join(out, f))

    if not os.path.exists(os.path.join(out, "events_left", "events.h5")):
        print("  unzip    events_left")
        zipfile.ZipFile(os.path.join(out, files[0])).extractall(os.path.join(out, "events_left"))
    if not os.path.exists(os.path.join(out, "flow")):
        print("  unzip    flow")
        zipfile.ZipFile(os.path.join(out, files[1])).extractall(os.path.join(out, "flow"))
    return out


def download_reblur(data_dir):
    out = os.path.join(data_dir, "reblur")
    os.makedirs(out, exist_ok=True)
    print(f"REBlur -> {os.path.relpath(out)}")
    for f in REBLUR_FILES:
        download(f"{REBLUR_URL}/{f}", os.path.join(out, f))
        marker = os.path.join(out, f".unzipped_{f}")   # the archives' internal layout differs, so mark by file
        if not os.path.exists(marker):
            print(f"  unzip    {f}")
            zipfile.ZipFile(os.path.join(out, f)).extractall(os.path.join(out, os.path.splitext(f)[0]))
            open(marker, "w").close()
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dataset", nargs="?", default="dsec", choices=["dsec", "reblur"],
                        help="which dataset to fetch (default: dsec)")
    parser.add_argument("--sequence", default="thun_00_a", help="DSEC training sequence name")
    parser.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "data"),
                        help="data folder (default: <repo>/data)")
    args = parser.parse_args()

    if args.dataset == "reblur":
        out = download_reblur(args.data_dir)
    else:
        out = download_dsec(args.sequence, args.data_dir)
    print("done:", sorted(os.listdir(out)))
