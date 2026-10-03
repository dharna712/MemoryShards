"""
Shared data helpers for the v2 face-embedding fine-tune: age-spanning
identities (AgeDB) plus CelebA, cached as aligned 160px MTCNN crops, and the
degradations (blur, low-res, JPEG, noise, low light) used both to train
robustness and to build the hard evaluation sets.

AgeDB layout after extraction (data/raw/agedb/...):
    <root>/id_0000_MariaCallas/0_MariaCallas_35_f.jpg    -> identity, age, sex
"""

import io
import os
import random
import re
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from PIL import Image, ImageFilter

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW = REPO_ROOT / "data" / "raw"
CACHE = RAW / "face_cache"
AGEDB_NAME = re.compile(r"^\d+_(?P<name>.+)_(?P<age>\d+)_(?P<sex>[fm])\.jpg$", re.IGNORECASE)

HOLDOUT_AGEDB_IDENTITIES = 60  # identities never trained on, used for evaluation
SEED = 7


def scan_agedb():
    """-> {identity: [(path, age), ...]} sorted by identity folder name."""
    root = next(RAW.joinpath("agedb").rglob("id_0000_*")).parent
    out = {}
    for d in sorted(root.glob("id_*")):
        items = []
        for p in d.glob("*.jpg"):
            m = AGEDB_NAME.match(p.name)
            if m:
                items.append((str(p), int(m.group("age"))))
        if items:
            out[d.name] = items
    return out


def split_agedb(agedb):
    ids = sorted(agedb)
    rng = random.Random(SEED)
    rng.shuffle(ids)
    held = set(ids[:HOLDOUT_AGEDB_IDENTITIES])
    return ({i: agedb[i] for i in ids if i not in held},
            {i: agedb[i] for i in ids if i in held})


def detect_crop(mtcnn, img):
    """160x160 raw 0-255 uint8 crop (matches training/inference: no standardization)."""
    face = mtcnn(img)
    if face is None:
        img = img.resize((160, 160), Image.BILINEAR)
        return torch.from_numpy(np.asarray(img)).permute(2, 0, 1).contiguous().byte()
    return face.clamp(0, 255).byte()


def cached_crop(mtcnn, path):
    key = CACHE / (re.sub(r"[^A-Za-z0-9]+", "_", str(Path(path).relative_to(RAW))) + ".pt")
    if key.exists():
        return torch.load(key)
    crop = detect_crop(mtcnn, Image.open(path).convert("RGB"))
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save(crop, key)
    return crop


def _to_pil(t):
    return Image.fromarray(t.permute(1, 2, 0).numpy())


def _to_tensor(img):
    return torch.from_numpy(np.asarray(img.convert("RGB"))).permute(2, 0, 1).contiguous().float()


def degrade(crop_u8, rng=random):
    """One random real-world degradation (or a stack of two) on a 160px crop."""
    img = _to_pil(crop_u8)
    kinds = ["blur", "lowres", "jpeg", "noise", "dark", "motion"]
    for k in rng.sample(kinds, rng.choice([1, 1, 2])):
        if k == "blur":
            img = img.filter(ImageFilter.GaussianBlur(rng.uniform(1.5, 4.0)))
        elif k == "lowres":
            s = rng.choice([24, 32, 40, 56])
            img = img.resize((s, s), Image.BILINEAR).resize((160, 160), Image.BILINEAR)
        elif k == "jpeg":
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=rng.randint(8, 30))
            img = Image.open(buf).convert("RGB")
        elif k == "noise":
            a = np.asarray(img).astype(np.float32)
            a += np.random.normal(0, rng.uniform(8, 25), a.shape)
            img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        elif k == "dark":
            a = np.asarray(img).astype(np.float32) / 255.0
            a = np.clip((a ** rng.uniform(1.6, 2.6)) * rng.uniform(0.45, 0.8), 0, 1)
            img = Image.fromarray((a * 255).astype(np.uint8))
        elif k == "motion":
            n = rng.choice([7, 9, 13])
            a = np.asarray(img).astype(np.float32)
            axis = rng.choice([0, 1])
            acc = sum(np.roll(a, s - n // 2, axis=axis) for s in range(n)) / n
            img = Image.fromarray(acc.astype(np.uint8))
    return _to_tensor(img)


def plain(crop_u8):
    return crop_u8.float()


def light_aug(crop_u8, rng=random):
    """Ordinary photometric jitter + flip for clean views."""
    t = crop_u8.float()
    if rng.random() < 0.5:
        t = torch.flip(t, dims=[2])
    t = (t * rng.uniform(0.85, 1.15) + rng.uniform(-12, 12)).clamp(0, 255)
    return t
