"""
Group a folder of photos by person, the way a phone gallery's "People" view
does. A thin command-line wrapper over recognize_people.py, so the CLI and the
web app always use the same detection and clustering.

Prints counts and filenames only. It never writes face crops to disk; if you
run it on real personal photos, keep the output private.

Usage:
    python src/cluster_own_photos.py --photos_dir "C:/path/to/photos"
    python src/cluster_own_photos.py --photos_dir ./photos --checkpoint checkpoints/face_embedding_head.pt
"""

import argparse
import os
from pathlib import Path

import recognize_people as rp

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--photos_dir", required=True, help="folder of photos to group")
    parser.add_argument("--checkpoint", default=str(rp.CHECKPOINT))
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    rp.CHECKPOINT = Path(args.checkpoint)

    photos = sorted(p for p in Path(args.photos_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    print(f"[1/2] found {len(photos)} photo(s) in {args.photos_dir}")
    if not photos:
        return

    people, photo_to_people = rp.recognize_people(photos)
    print(f"[2/2] {len(people)} people in {len(photo_to_people)}/{len(photos)} photos\n")

    by_person = {}
    for photo, ids in photo_to_people.items():
        for pid in ids:
            by_person.setdefault(pid, []).append(Path(photo).name)
    for person in people:
        names = sorted(by_person.get(person["id"], []))
        print(f"{person['label']}: {person['face_count']} faces in {person['photo_count']} photos")
        for name in names[:10]:
            print(f"    {name}")
        if len(names) > 10:
            print(f"    ... +{len(names) - 10} more")


if __name__ == "__main__":
    main()
