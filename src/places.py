"""
Place names for events.

Offline (default, nothing leaves the machine): reverse_geocoder returns the
nearest *listed town*, which is fine for city-scale photos but wrong at
neighbourhood scale (Baner comes back as "Khadki", the Gateway of India as
"Uran", 30 km away). So the distance to that town is checked and the label is
hedged accordingly: a name when it is close, "near <town>" when it is a few
km away, and only the region when no listed town is anywhere close.

Online (opt-in per request): OpenStreetMap Nominatim returns the suburb or
neighbourhood ("Kothrud, Pune"). It sends the event's coordinates to a third
party, so it is only used when the user asks for it, once per event, at
Nominatim's 1 request/second limit, with a fallback to the offline label.
"""

import json
import time
import urllib.parse
import urllib.request
from math import asin, cos, radians, sin, sqrt

import reverse_geocoder as rg

NAME_KM = 6       # nearest town this close: use its name
NEAR_KM = 25      # this close: "near <town>"; farther: region only
USER_AGENT = "MemoryShards-college-project/1.0 (educational)"

_online_cache = {}


def _km(a, b):
    lat1, lon1, lat2, lon2 = map(radians, [*a, *b])
    h = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * asin(sqrt(h))


def offline_place(lat, lon):
    r = rg.search([(lat, lon)], mode=1)[0]
    dist = _km((lat, lon), (float(r["lat"]), float(r["lon"])))
    region = r["admin1"] or r["admin2"] or ""
    if dist <= NAME_KM:
        name = r["name"]
    elif dist <= NEAR_KM:
        name = "near " + r["name"]
    else:
        name = region or r["name"]
    parts = [name]
    if region and region != name:
        parts.append(region)
    parts.append(r["cc"])
    return ", ".join(p for p in parts if p)


def online_place(lat, lon):
    """Suburb-level name from OpenStreetMap, or None if unavailable."""
    key = (round(lat, 3), round(lon, 3))
    if key in _online_cache:
        return _online_cache[key]
    url = "https://nominatim.openstreetmap.org/reverse?" + urllib.parse.urlencode(
        {"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 16, "addressdetails": 1})
    label = None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        addr = json.load(urllib.request.urlopen(req, timeout=5)).get("address", {})
        local = (addr.get("suburb") or addr.get("neighbourhood") or addr.get("city_district")
                 or addr.get("quarter") or addr.get("village") or addr.get("hamlet"))
        city = addr.get("city") or addr.get("town") or addr.get("municipality") or addr.get("county")
        parts = [p for p in (local, city if city != local else None) if p]
        if parts:
            label = ", ".join(parts + [(addr.get("country_code") or "").upper()]).rstrip(", ")
    except Exception:
        label = None
    _online_cache[key] = label
    time.sleep(1.1)  # Nominatim usage policy: at most 1 request per second
    return label


def place_label(lat, lon, online=False):
    if online:
        label = online_place(lat, lon)
        if label:
            return label
    return offline_place(lat, lon)
