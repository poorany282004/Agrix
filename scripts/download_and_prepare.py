# agrix/scripts/download_and_prepare.py
import random
from pathlib import Path
from PIL import Image

# Paths
RAW = Path(__file__).resolve().parents[1] / "dataset" / "raw"
OUT = Path(__file__).resolve().parents[1] / "dataset" / "merged"

# Train/val split
SPLIT = 0.8

# Resize dimensions
SIZE = (224, 224)

# Allowed image extensions
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Map many possible labels -> canonical labels
CANONICAL_MAP = {
    "blast": "blast",
    "rice_blast": "blast",
    "leaf_blast": "blast",
    "brown_spot": "brown_spot",
    "brownspot": "brown_spot",
    "bacterial_blight": "bacterial_blight",
    "bacterial leaf blight": "bacterial_blight",
    "healthy": "healthy",
    "normal": "healthy",
    "no_disease": "healthy"
}

def find_images(root):
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            yield p, p.parent.name.lower()

def map_label(label):
    lab = label.strip().lower().replace(" ", "_").replace("-", "_")
    if lab in CANONICAL_MAP:
        return CANONICAL_MAP[lab]
    for k in CANONICAL_MAP:
        if k in lab:
            return CANONICAL_MAP[k]
    return None

def copy_resize(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        im = Image.open(src).convert("RGB")
        im = im.resize(SIZE, Image.BILINEAR)
        im.save(dst, quality=92)
    except Exception as e:
        print("skip", src, e)

def build():
    OUT.mkdir(parents=True, exist_ok=True)
    images = []
    for p, label in find_images(RAW):
        cl = map_label(label)
        if cl:
            images.append((p, cl))
    print("Mapped images:", len(images))

    from collections import defaultdict
    bycls = defaultdict(list)
    for p, c in images:
        bycls[c].append(p)

    for c, lst in bycls.items():
        random.shuffle(lst)
        cut = int(len(lst) * SPLIT)
        for i, p in enumerate(lst):
            part = "train" if i < cut else "val"
            dst = OUT / part / c / p.name
            copy_resize(p, dst)

    print("Merged dataset written to", OUT)

if __name__ == "__main__":
    build()
