"""
Pull children (and a smaller set of very old faces) from UTKFace, which has
age labels but NO identity labels. These images are used as extra
self-supervised identities in training: each image is its own "identity",
the positive is a differently augmented/degraded view of the same photo, and
every other image in the batch is a negative. That teaches the embedding to
tell different children apart (the weak spot of a model trained on adult
celebrities) without needing identity labels.

Output: data/raw/utkface/<kid|old>/<n>_<age>.jpg
"""

import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from datasets import load_dataset

OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "utkface"
MAX_KIDS = 1600
MAX_OLD = 500
KID_MAX_AGE = 12
OLD_MIN_AGE = 70


def main():
    (OUT / "kid").mkdir(parents=True, exist_ok=True)
    (OUT / "old").mkdir(parents=True, exist_ok=True)
    ds = load_dataset("nu-delta/utkface", split="train", streaming=True)
    kids = old = 0
    for ex in ds:
        age = ex["age"]
        if age <= KID_MAX_AGE and kids < MAX_KIDS:
            ex["image"].convert("RGB").save(OUT / "kid" / f"{kids}_{age}.jpg", "JPEG", quality=95)
            kids += 1
        elif age >= OLD_MIN_AGE and old < MAX_OLD:
            ex["image"].convert("RGB").save(OUT / "old" / f"{old}_{age}.jpg", "JPEG", quality=95)
            old += 1
        if kids >= MAX_KIDS and old >= MAX_OLD:
            break
    print(f"saved {kids} child faces (age<={KID_MAX_AGE}) and {old} old faces (age>={OLD_MIN_AGE})")


if __name__ == "__main__":
    main()
