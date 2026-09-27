"""
Real geotagged, timestamped test photos for the four messy-geography
scenarios Harsh asked about: photos taken close together in one city,
across a city's neighbourhoods, across cities in one state, and across
states/countries. Same idea and same source as
download_wikimedia_geotagged.py (real Wikimedia Commons files, real
embedded EXIF, no synthetic coordinates) — just anchored on named points
instead of topic keywords, using Commons' geosearch so results actually
cluster the way each scenario needs.

We never touch social media accounts (Instagram/Facebook/etc.) for this:
scraping those violates their terms and personal, non-consented photos of
strangers is a different privacy problem than CC-licensed Commons uploads.

Output:
    data/raw/scenario_<name>/<n>.jpg
    data/raw/scenario_<name>/manifest.csv  (path, title, source_url, lat, lon)

Usage:
    python src/download_scenario_photos.py [scenario ...]   # default: all
"""

import csv
import io
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image

API_URL = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "MemoryShardsResearch/1.0 (student project; contact: nightfuryharsh@gmail.com)"}
OUT_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"

# name -> (label, [(point_label, lat, lon), ...], radius_m)
# radius is Commons geosearch's own cap (10000 m); scenarios differ by how
# far apart the *points* are, not the per-point radius.
SCENARIOS = {
    "same_neighbourhood": (
        "Multiple points ~1-2km apart within one Pune neighbourhood (Kothrud)",
        [("kothrud_a", 18.5074, 73.8077), ("kothrud_b", 18.5001, 73.8177), ("kothrud_c", 18.5140, 73.8210)],
        3000,
    ),
    "same_city": (
        "Different neighbourhoods of the same city, ~5-12km apart (Pune: Kothrud vs Baner vs Koregaon Park)",
        [("kothrud", 18.5074, 73.8077), ("baner", 18.5590, 73.7868), ("koregaon_park", 18.5362, 73.8938)],
        4000,
    ),
    "same_state": (
        "Different cities in Maharashtra, ~120-250km apart (Mumbai, Pune, Nashik)",
        [("mumbai", 19.0760, 72.8777), ("pune", 18.5204, 73.8567), ("nashik", 19.9975, 73.7898)],
        8000,
    ),
    "cross_state": (
        "Different states/regions, >800km apart (Delhi, Bengaluru, Kolkata)",
        [("delhi", 28.6139, 77.2090), ("bengaluru", 12.9716, 77.5946), ("kolkata", 22.5726, 88.3639)],
        8000,
    ),
}


def geosearch(lat, lon, radius_m, limit=20):
    resp = requests.get(API_URL, params={
        "action": "query", "list": "geosearch",
        "gscoord": f"{lat}|{lon}", "gsradius": radius_m, "gsnamespace": 6, "gslimit": limit,
        "format": "json",
    }, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json().get("query", {}).get("geosearch", [])


def imageinfo(titles):
    if not titles:
        return {}
    resp = requests.get(API_URL, params={
        "action": "query", "titles": "|".join(titles), "prop": "imageinfo",
        "iiprop": "url|extmetadata", "format": "json",
    }, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})
    return {p["title"]: p.get("imageinfo", [{}])[0] for p in pages.values() if p.get("imageinfo")}


def has_real_exif_gps_and_time(content):
    """Commons' page metadata (checked in fetch_point) says a file has a
    location and a displayed date, but that's often manually entered by the
    uploader — plenty of "geotagged, dated" Commons files are scans or
    illustrations with no GPS/DateTimeOriginal actually embedded in the
    image. Re-parse the downloaded bytes the same way Extract will, and
    only trust files that carry the real tags."""
    try:
        exif = Image.open(io.BytesIO(content))._getexif()
    except Exception:
        return False
    if not exif:
        return False
    return 34853 in exif and 36867 in exif  # GPSInfo, DateTimeOriginal


def fetch_point(point_label, lat, lon, radius_m, per_point_limit=6, candidate_pool=40):
    """Download-and-verify until per_point_limit real photos are found, or
    the candidate pool from Commons' geosearch runs out."""
    found = []
    hits = geosearch(lat, lon, radius_m, limit=candidate_pool)
    if not hits:
        return found
    titles = [h["title"] for h in hits]
    info_by_title = imageinfo(titles)
    checked = 0
    for h in hits:
        if len(found) >= per_point_limit:
            break
        info = info_by_title.get(h["title"])
        if not info:
            continue
        meta = info.get("extmetadata", {})
        if "DateTimeOriginal" not in meta:
            continue
        url = info.get("url", "")
        ext = Path(urlparse(url).path).suffix.lower()
        if ext not in (".jpg", ".jpeg"):
            continue  # PNGs on Commons are almost never real camera output
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
        except requests.RequestException:
            continue
        checked += 1
        if not has_real_exif_gps_and_time(resp.content):
            continue
        found.append({"point": point_label, "title": h["title"], "url": url,
                       "lat": h["lat"], "lon": h["lon"], "content": resp.content})
    print(f"    {checked} candidates downloaded and checked, {len(found)} had real embedded GPS+timestamp")
    return found


def run_scenario(name):
    label, points, radius_m = SCENARIOS[name]
    print(f"\n[{name}] {label}")
    out_dir = OUT_ROOT / f"scenario_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, idx = [], 0

    for point_label, lat, lon in points:
        print(f"  searching near {point_label} ({lat}, {lon}), radius {radius_m}m...")
        try:
            items = fetch_point(point_label, lat, lon, radius_m)
        except requests.RequestException as e:
            print(f"    request failed: {e}")
            continue
        for item in items:
            ext = Path(urlparse(item["url"]).path).suffix.lower()
            filename = f"{idx}{ext}"
            (out_dir / filename).write_bytes(item["content"])
            rows.append({"path": filename, "point": item["point"], "title": item["title"],
                         "source_url": item["url"], "lat": item["lat"], "lon": item["lon"]})
            idx += 1
        time.sleep(0.5)  # be polite to the API between points

    manifest = out_dir / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "point", "title", "source_url", "lat", "lon"])
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {len(rows)} photos in {out_dir}")
    return len(rows)


def main():
    import sys
    names = [a for a in sys.argv[1:]] or list(SCENARIOS)
    total = 0
    for name in names:
        if name not in SCENARIOS:
            print(f"unknown scenario {name!r}, options: {list(SCENARIOS)}")
            continue
        total += run_scenario(name)
    print(f"\ndone: {total} photos total across {len(names)} scenario(s)")


if __name__ == "__main__":
    main()
