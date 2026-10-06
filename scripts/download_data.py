"""Download the DSEC files used by notebooks/02_cmax_optical_flow_dsec.ipynb.

Pulls only the left event camera (~300 MB) plus forward flow ground truth (~12 MB) for
one sequence (default thun_00_a) and unpacks them into:

    data/dsec/<sequence>/
        events_left/events.h5
        events_left/rectify_map.h5
        flow/*.png
        <sequence>_optical_flow_forward_timestamps.txt

Files that already exist are skipped, so it is safe to run again.

Usage (from anywhere):
    python scripts/download_data.py
    python scripts/download_data.py --sequence thun_00_a --data-dir data
"""

import argparse
import os
import subprocess
import zipfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_URL = "https://download.ifi.uzh.ch/rpg/DSEC/train"


def download(url, dest):
    if os.path.exists(dest):
        print(f"  have     {os.path.basename(dest)}")
        return
    print(f"  download {os.path.basename(dest)}")
    tmp = dest + ".part"  # so an interrupted download is never mistaken for a finished one
    subprocess.run(["curl", "-fL", "--progress-bar", "-o", tmp, url], check=True)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sequence", default="thun_00_a", help="DSEC training sequence name")
    parser.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "data"),
                        help="data folder (default: <repo>/data)")
    args = parser.parse_args()

    out = download_dsec(args.sequence, args.data_dir)
    print("done:", sorted(os.listdir(out)))
