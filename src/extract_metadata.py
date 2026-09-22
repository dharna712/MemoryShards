"""
Day 3 — the Extract stage: pull timestamp + GPS out of a photo's own
embedded EXIF (not from any side-channel metadata), then reverse-geocode
the coordinates to a place name.

This is deliberately dumb and classical — no model, no learning. It's the
foundation the rest of the pipeline sits on, and it's designed to work the
same way on Wikimedia photos (used here to validate it) and personal
photos (used later, in the real demo).
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

import reverse_geocoder as rg
from PIL import Image

# Windows consoles default to cp1252, which can't print many photo titles
# (accented/non-Latin characters are common on Commons) — force UTF-8 for
# stdout so a display encoding issue doesn't look like an extraction bug
sys.stdout.reconfigure(encoding="utf-8")

GPS_IFD_TAG = 0x8825
DATETIME_TAG = 0x0132
DATETIME_ORIGINAL_TAG = 0x9003  # lives in the Exif SubIFD, not the top-level IFD
EXIF_IFD_TAG = 0x8769


def _dms_to_decimal(dms, ref):
    degrees, minutes, seconds = dms
    decimal = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


def extract_gps(exif):
    """Returns (lat, lon) or None if no embedded GPS EXIF."""
    gps_ifd = exif.get_ifd(GPS_IFD_TAG) if GPS_IFD_TAG in exif else None
    if not gps_ifd:
        return None
    try:
        lat = _dms_to_decimal(gps_ifd[2], gps_ifd[1])  # tag 2=GPSLatitude, 1=GPSLatitudeRef
        lon = _dms_to_decimal(gps_ifd[4], gps_ifd[3])  # tag 4=GPSLongitude, 3=GPSLongitudeRef
        return (lat, lon)
    except (KeyError, TypeError, ZeroDivisionError):
        return None


def extract_timestamp(exif):
    """Returns a datetime or None. Prefers DateTimeOriginal (Exif SubIFD)
    over the top-level DateTime tag, since DateTimeOriginal is when the
    photo was actually taken (DateTime can be a later file-modified time)."""
    exif_sub = exif.get_ifd(EXIF_IFD_TAG) if EXIF_IFD_TAG in exif else {}
    raw = exif_sub.get(DATETIME_ORIGINAL_TAG) or exif.get(DATETIME_TAG)
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def extract_photo_metadata(path):
    """Returns {'timestamp': datetime|None, 'lat': float|None, 'lon': float|None,
    'place': str|None} for one photo, reading only its own embedded EXIF."""
    img = Image.open(path)
    exif = img.getexif()

    timestamp = extract_timestamp(exif)
    gps = extract_gps(exif)

    place = None
    if gps:
        # reverse_geocoder is offline (bundled city/country lookup table) —
        # no API calls, no rate limits, works without internet.
        # mode=1 forces single-threaded lookup — its default (mode=2) spawns
        # worker processes on every call, which cost 3-5s of overhead per
        # photo when profiled, dwarfing the actual KD-tree query time for a
        # single coordinate.
        result = rg.search([gps], mode=1)[0]
        place = f"{result['name']}, {result['admin1']}, {result['cc']}"

    return {
        "path": str(path),
        "timestamp": timestamp,
        "lat": gps[0] if gps else None,
        "lon": gps[1] if gps else None,
        "place": place,
    }


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data" / "raw" / "wikimedia_geotagged"
    with open(data_dir / "manifest.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    results = []
    for row in rows:
        meta = extract_photo_metadata(data_dir / row["path"])
        meta["title"] = row["title"]
        results.append(meta)

    ok = [r for r in results if r["timestamp"] and r["place"]]
    print(f"extracted usable (timestamp + place) metadata for {len(ok)}/{len(results)} photos\n")
    for r in sorted(ok, key=lambda r: r["timestamp"])[:8]:
        print(f"  {r['timestamp']}  {r['place']:<30}  {r['title']}")

    return results


if __name__ == "__main__":
    main()
