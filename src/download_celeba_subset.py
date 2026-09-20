"""
Day-2: download a small, real subset of CelebA (public, licensed, labeled
identities) for fine-tuning the face-embedding model.

We deliberately do NOT pull the full ~200K-image dataset — training on a
few hundred identities is enough to prove the model learns real face
similarity, and it keeps download size and training time sane for a
solo, two-week timeline.

Source: the `flwrlabs/celeba` mirror on the Hugging Face Hub, which serves
the official CelebA images + `celeb_id` identity labels (no Google Drive/
Kaggle auth needed). Rows are grouped by identity, so we can stream just
the first N identity blocks instead of downloading the whole dataset.

Output layout:
    data/raw/celeba_subset/<celeb_id>/<n>.jpg
    data/raw/celeba_subset/manifest.csv   (path, celeb_id)
"""

import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from datasets import load_dataset

TARGET_IDENTITIES = 120       # how many distinct people to collect
MAX_IMAGES_PER_IDENTITY = 20  # cap per person (avoids one huge folder)
MIN_IMAGES_PER_IDENTITY = 10  # drop people with too few images for triplets
MAX_ROWS_TO_SCAN = 20000      # safety cap so a bad assumption can't run forever

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "celeba_subset"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[1/3] streaming flwrlabs/celeba (grouped by identity)...")

    ds = load_dataset("flwrlabs/celeba", split="train", streaming=True)

    per_identity_count = defaultdict(int)
    manifest_rows = []
    identities_seen = []
    rows_scanned = 0

    for example in ds:
        rows_scanned += 1
        if rows_scanned > MAX_ROWS_TO_SCAN:
            print("  hit MAX_ROWS_TO_SCAN safety cap, stopping")
            break

        cid = example["celeb_id"]
        if cid not in identities_seen:
            if len(identities_seen) >= TARGET_IDENTITIES:
                # we've moved on to a new identity beyond our target — done
                break
            identities_seen.append(cid)

        if per_identity_count[cid] >= MAX_IMAGES_PER_IDENTITY:
            continue

        idx = per_identity_count[cid]
        person_dir = OUT_DIR / str(cid)
        person_dir.mkdir(exist_ok=True)
        img_path = person_dir / f"{idx}.jpg"
        example["image"].convert("RGB").save(img_path, "JPEG", quality=92)

        manifest_rows.append((img_path.relative_to(OUT_DIR).as_posix(), cid))
        per_identity_count[cid] += 1

    print(f"[2/3] scanned {rows_scanned} rows, saw {len(identities_seen)} identities")

    # drop identities with too few images (not enough for anchor/positive pairs)
    kept_rows = [r for r in manifest_rows if per_identity_count[r[1]] >= MIN_IMAGES_PER_IDENTITY]
    dropped = len(manifest_rows) - len(kept_rows)

    manifest_path = OUT_DIR / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "celeb_id"])
        writer.writerows(kept_rows)

    kept_identities = {cid for _, cid in kept_rows}
    print(f"[3/3] kept {len(kept_rows)} images across {len(kept_identities)} identities "
          f"(dropped {dropped} images from identities with < {MIN_IMAGES_PER_IDENTITY} photos)")
    print(f"\nmanifest written to {manifest_path}")


if __name__ == "__main__":
    main()
