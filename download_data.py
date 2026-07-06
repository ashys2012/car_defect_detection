"""
Download the CarDD dataset (YOLO-format mirror of the official dataset) via
kagglehub, and copy it into ./data/ so it's self-contained in the repo rather
than depending on kagglehub's cache under ~/.cache - which is not guaranteed
to survive (in this project's dev sandbox, ~/.cache got wiped by an unrelated
session-boundary cleanup partway through, which is what prompted this).

Requires a Kaggle API token: create one at kaggle.com -> Settings -> API ->
"Create New Token", then either:
  - place the downloaded kaggle.json at ~/.kaggle/kaggle.json, or
  - export KAGGLE_API_TOKEN=<token> (Kaggle's newer token-based auth,
    supported by kagglehub).

Usage:
    source .venv/bin/activate
    python download_data.py
"""

import shutil
from pathlib import Path

import kagglehub

DATASET = "gabrielfcarvalho/cardd-with-yolo-annotations-images-labels"
LOCAL_DATA_DIR = Path("data")

DATA_YAML_TEMPLATE = """\
# CarDD (Car Damage Detection), YOLO-format mirror of the official dataset.
# Split sizes (train/val/test = 2816/810/374) match the official CarDD paper's
# published split exactly - this is the authors' original split, not a fresh
# reshuffle, which avoids leakage from re-splitting an already-curated dataset.
path: {root}
train: train/images
val: val/images
test: test/images

nc: 6
names: ["dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat"]
"""


def main():
    cache_path = Path(kagglehub.dataset_download(DATASET))
    print(f"Downloaded to cache: {cache_path}")

    LOCAL_DATA_DIR.mkdir(exist_ok=True)
    for split in ("train", "val", "test"):
        dest = LOCAL_DATA_DIR / split
        if dest.exists():
            print(f"{dest} already present, skipping copy")
            continue
        shutil.copytree(cache_path / split, dest)
        print(f"Copied {split} -> {dest}")

    out = LOCAL_DATA_DIR / "cardd.yaml"
    out.write_text(DATA_YAML_TEMPLATE.format(root=LOCAL_DATA_DIR.resolve()))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
