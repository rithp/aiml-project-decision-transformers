"""
fetch_datasets.py  —  download D4RL pre-processed datasets using only stdlib.
Run from: decision-transformer/gym/data/

Usage:
    python3 fetch_datasets.py                      # downloads all 9 datasets
    python3 fetch_datasets.py hopper medium        # downloads only hopper-medium-v2
"""

import sys
import urllib.request
import os

BASE_URL = "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2"

DATASETS = [
    ("hopper",       "medium"),
    ("hopper",       "medium-replay"),
    ("hopper",       "expert"),
    ("halfcheetah",  "medium"),
    ("halfcheetah",  "medium-replay"),
    ("halfcheetah",  "expert"),
    ("walker2d",     "medium"),
    ("walker2d",     "medium-replay"),
    ("walker2d",     "expert"),
]


def download(env, dataset_type, dest_dir="."):
    filename = f"{env}_{dataset_type}-v2.pkl"
    url = f"{BASE_URL}/{filename}"
    out_path = os.path.join(dest_dir, f"{env}-{dataset_type}-v2.pkl")

    if os.path.exists(out_path):
        print(f"[skip] {out_path} already exists")
        return

    print(f"[download] {url}")
    print(f"        → {out_path}")

    def progress(count, block_size, total_size):
        if total_size > 0:
            pct = min(count * block_size / total_size * 100, 100)
            mb = total_size / 1e6
            print(f"\r  {pct:.1f}%  of {mb:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, out_path, reporthook=progress)
    print()  # newline after progress
    print(f"  saved → {out_path}")


if __name__ == "__main__":
    dest = os.path.dirname(os.path.abspath(__file__))

    if len(sys.argv) == 3:
        # single dataset
        download(sys.argv[1], sys.argv[2], dest_dir=dest)
    else:
        # all datasets
        for env, dtype in DATASETS:
            download(env, dtype, dest_dir=dest)
