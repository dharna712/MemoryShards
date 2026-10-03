"""
Download the aligned face crops of YLFW (Young Labeled Faces in the Wild:
identity-labelled photos of children, internet images of young celebrities /
actors) from the Hugging Face mirror hieupth/ylfw. Research dataset:
training/eval use only, data/ stays gitignored, nothing is redistributed.

The repo is ~10k tiny files; fetched one at a time that is hours of pure
request latency, so this pulls them with many parallel connections.

Output: data/raw/ylfw/<Race_id>/<file>.png   (folder name = identity)
"""

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "hieupth/ylfw"
OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "ylfw"
WORKERS = 24


def list_files():
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}", timeout=60) as r:
        info = json.load(r)
    return [s["rfilename"] for s in info["siblings"] if s["rfilename"].startswith("benchmark/aligned/")]


def fetch(rel):
    dest = OUT / Path(rel).relative_to("benchmark/aligned")
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{rel}"
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                dest.write_bytes(r.read())
            return True
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return False


def main():
    files = list_files()
    print(f"{len(files)} aligned files to fetch with {WORKERS} connections")
    with ThreadPoolExecutor(WORKERS) as ex:
        ok = sum(ex.map(fetch, files))
    ids = len([d for d in OUT.iterdir() if d.is_dir()])
    print(f"saved {ok}/{len(files)} images across {ids} child identities")


if __name__ == "__main__":
    main()
