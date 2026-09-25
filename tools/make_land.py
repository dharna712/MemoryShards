"""
Builds web/media/land.json, the coastline data for the results map.

Source: Natural Earth 110m land polygons (public domain,
https://www.naturalearthdata.com). Each ring is stored as delta-encoded
integers in tenths of a degree, [lon0, lat0, dlon, dlat, ...], to keep the file
small; web/try.html decodes it.

Usage:
    python tools/make_land.py path/to/ne_110m_land.geojson
"""

import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
out = Path(__file__).resolve().parent.parent / "web" / "media" / "land.json"

rings = []
for feature in json.loads(src.read_text(encoding="utf-8"))["features"]:
    geom = feature["geometry"]
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    for poly in polys:
        ring = poly[0]  # outer ring only; holes don't matter at this scale
        pts = [(round(lon * 10), round(lat * 10)) for lon, lat in ring]
        flat = [pts[0][0], pts[0][1]]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            flat += [x1 - x0, y1 - y0]
        rings.append(flat)

out.write_text(json.dumps(rings, separators=(",", ":")), encoding="utf-8")
print(f"{len(rings)} rings, {out.stat().st_size / 1024:.0f} KB -> {out}")
