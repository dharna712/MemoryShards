"""
Day 3: download a small set of real geotagged, timestamped travel photos
from Wikimedia Commons — the public training/dev data for the classical
(non-DL) part of the pipeline: EXIF extraction, reverse geocoding, DBSCAN
event clustering.

Wikimedia Commons doesn't have one clean "geotagged photos" category we
can page through (checked — the obvious category names don't exist).
Instead: search across a handful of travel/event-shaped topics, and keep
only files whose Commons metadata confirms they have both a GPS location
and a DateTimeOriginal — the same two fields the Extract stage will later
pull from the file's own embedded EXIF.

We download the actual image bytes (not just the API's metadata) so the
Extract stage genuinely re-parses EXIF from a real file, the same way it
will for personal photos later — the Commons metadata here is only used
to *find* usable photos, not as a shortcut around real extraction.

Output:
    data/raw/wikimedia_geotagged/<n>.jpg
    data/raw/wikimedia_geotagged/manifest.csv  (path, title, source_url)
"""

import csv
import os
from pathlib import Path
from urllib.parse import urlparse

import requests

API_URL = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "MemoryShardsResearch/1.0 (student project; contact: nightfuryharsh@gmail.com)"}

# diverse topics so events don't all look alike once we get to clustering/captioning
SEARCH_TERMS = [
    "beach", "temple", "mountain hiking", "street market", "waterfall",
    "old town square", "harbor sunset", "railway station", "cafe terrace",
    "national park",
]
IMAGES_PER_TERM = 15
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "wikimedia_geotagged"


def search_geotagged_files(term, limit):
    resp = requests.get(API_URL, params={
        "action": "query",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {term}",
        "gsrnamespace": 6,
        "gsrlimit": limit * 3,  # over-fetch, since not all results have coords+date
        "prop": "coordinates|imageinfo",
        "iiprop": "url|extmetadata",
        "format": "json",
    }, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})

    results = []
    for page in pages.values():
        if "coordinates" not in page or not page.get("imageinfo"):
            continue
        info = page["imageinfo"][0]
        meta = info.get("extmetadata", {})
        if "DateTimeOriginal" not in meta:
            continue
        results.append({
            "title": page["title"],
            "url": info["url"],
            "lat": page["coordinates"][0]["lat"],
            "lon": page["coordinates"][0]["lon"],
        })
        if len(results) >= limit:
            break
    return results


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    idx = 0

    for term in SEARCH_TERMS:
        print(f"[search] {term!r}...")
        try:
            found = search_geotagged_files(term, IMAGES_PER_TERM)
        except requests.RequestException as e:
            print(f"  request failed: {e}")
            continue
        print(f"  {len(found)} usable (geotagged + dated) results")

        for item in found:
            # URLs carry tracking query params (?utm_source=...) — strip
            # them before reading the extension, or Path().suffix grabs
            # the tail of the query string instead of the real extension
            url_path = urlparse(item["url"]).path
            ext = Path(url_path).suffix.lower()
            if ext not in (".jpg", ".jpeg", ".png"):
                continue
            try:
                img_resp = requests.get(item["url"], headers=HEADERS, timeout=30)
                img_resp.raise_for_status()
            except requests.RequestException as e:
                print(f"    download failed for {item['title']}: {e}")
                continue

            filename = f"{idx}{ext}"
            (OUT_DIR / filename).write_bytes(img_resp.content)
            manifest_rows.append({
                "path": filename,
                "title": item["title"],
                "source_url": item["url"],
                "commons_lat": item["lat"],
                "commons_lon": item["lon"],
            })
            idx += 1

    manifest_path = OUT_DIR / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "title", "source_url", "commons_lat", "commons_lon"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\ndownloaded {len(manifest_rows)} images -> {OUT_DIR}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
