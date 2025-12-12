"""
test_adsb_route_corridor.py

Route-corridor ADS-B query using adsb.lol:
- Sample points along a route (great-circle approximation)
- Query aircraft near each sample point
- Deduplicate and rank by distance to route line

Defaults: KMSP -> KDEN (edit coords as needed)
"""

import os
import math
import time
import requests
from pprint import pprint

BASE = os.getenv("ADSBLOL_BASE", "https://api.adsb.lol").strip().rstrip("/")

# Default route: MSP -> DEN (approx)
ORIG = (float(os.getenv("ORIG_LAT", "44.8848")), float(os.getenv("ORIG_LON", "-93.2223")))   # KMSP
DEST = (float(os.getenv("DEST_LAT", "39.8561")), float(os.getenv("DEST_LON", "-104.6737")))  # KDEN

# Corridor settings
SAMPLES = int(os.getenv("ROUTE_SAMPLES", "7"))          # number of sample points along the route
QUERY_RADIUS_NM = float(os.getenv("QUERY_RADIUS_NM", "45"))  # radius around each sample point
MIN_ALT_FT = float(os.getenv("MIN_ALT_FT", "10000"))    # airborne filter

# Output
MAX_PRINT = int(os.getenv("MAX_PRINT", "25"))


def nm_to_km(nm: float) -> float:
    return nm * 1.852


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def interpolate_points(lat1, lon1, lat2, lon2, n: int):
    """
    Simple lat/lon interpolation. For corridors this is good enough.
    Returns n points including endpoints.
    """
    pts = []
    for i in range(n):
        t = 0 if n == 1 else i / (n - 1)
        lat = lat1 + (lat2 - lat1) * t
        lon = lon1 + (lon2 - lon1) * t
        pts.append((lat, lon))
    return pts


def fetch_near(lat: float, lon: float, dist_km: float) -> dict:
    url = f"{BASE}/v2/lat/{lat}/lon/{lon}/dist/{dist_km}"
    resp = requests.get(url, timeout=30, headers={"User-Agent": "AvWeather-RouteCorridor/1.0"})
    resp.raise_for_status()
    return resp.json()


def pick_callsign(obj: dict) -> str | None:
    for k in ("flight", "call", "callsign", "cs", "ident"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def normalize(obj: dict) -> dict:
    return {
        "hex": obj.get("hex") or obj.get("icao") or obj.get("icao24"),
        "callsign": pick_callsign(obj),
        "lat": obj.get("lat"),
        "lon": obj.get("lon"),
        "alt_baro_ft": obj.get("alt_baro", obj.get("alt")),
        "alt_geom_ft": obj.get("alt_geom", obj.get("galt")),
        "gs_kt": obj.get("gs", obj.get("spd")),
        "track_deg": obj.get("track", obj.get("dir")),
        "vr_fpm": obj.get("baro_rate", obj.get("vsi")),
        "type": obj.get("t"),
        "reg": obj.get("r"),
        "squawk": obj.get("squawk", obj.get("sqk")),
        "seen_s": obj.get("seen"),
        "seen_pos_s": obj.get("seen_pos"),
        "raw": obj,
    }


def alt_ft(ac: dict) -> float:
    for k in ("alt_baro_ft", "alt_geom_ft"):
        v = ac.get(k)
        if v is None:
            continue
        try:
            return float(v)
        except Exception:
            continue
    return -1.0


def min_distance_to_samples(ac: dict, samples: list[tuple[float, float]]) -> float:
    lat, lon = ac["lat"], ac["lon"]
    return min(haversine_km(lat, lon, s_lat, s_lon) for s_lat, s_lon in samples)


def main():
    print("=== ADS-B Route Corridor Test (adsb.lol) ===")
    print(f"Base: {BASE}")
    print(f"ORIG: {ORIG}  DEST: {DEST}")
    print(f"SAMPLES={SAMPLES}, QUERY_RADIUS_NM={QUERY_RADIUS_NM}, MIN_ALT_FT={MIN_ALT_FT}")

    route_samples = interpolate_points(ORIG[0], ORIG[1], DEST[0], DEST[1], SAMPLES)
    dist_km = nm_to_km(QUERY_RADIUS_NM)

    all_raw = []
    t0 = time.time()

    for idx, (lat, lon) in enumerate(route_samples, 1):
        print(f"Query {idx}/{len(route_samples)} at ({lat:.4f}, {lon:.4f}) dist_km={dist_km:.1f}")
        payload = fetch_near(lat, lon, dist_km)
        raw_list = payload.get("ac") or payload.get("aircraft") or payload.get("planes") or []
        all_raw.extend(raw_list)
        time.sleep(0.15)  # be polite

    dt = time.time() - t0
    print(f"\nFetched {len(all_raw)} raw aircraft rows across samples in {dt:.2f}s")

    # Normalize + dedupe by hex
    by_hex: dict[str, dict] = {}
    for obj in all_raw:
        ac = normalize(obj)
        hx = ac.get("hex")
        if not hx:
            continue
        if ac["lat"] is None or ac["lon"] is None:
            continue
        if alt_ft(ac) < MIN_ALT_FT:
            continue
        # keep the most recently seen_pos (smaller is newer)
        prev = by_hex.get(hx)
        if prev is None:
            by_hex[hx] = ac
        else:
            prev_seen = prev.get("seen_pos_s")
            cur_seen = ac.get("seen_pos_s")
            if prev_seen is None or (cur_seen is not None and cur_seen < prev_seen):
                by_hex[hx] = ac

    aircraft = list(by_hex.values())
    print(f"After dedupe + filters: {len(aircraft)} aircraft")

    # Rank by closeness to route samples
    for ac in aircraft:
        ac["dist_to_route_km"] = round(min_distance_to_samples(ac, route_samples), 2)

    aircraft.sort(key=lambda a: a["dist_to_route_km"])

    print(f"\nTop {min(MAX_PRINT, len(aircraft))} aircraft closest to corridor:\n")
    for ac in aircraft[:MAX_PRINT]:
        pprint({
            "callsign": ac["callsign"],
            "hex": ac["hex"],
            "type": ac["type"],
            "reg": ac["reg"],
            "alt_baro_ft": ac["alt_baro_ft"],
            "gs_kt": ac["gs_kt"],
            "track_deg": ac["track_deg"],
            "lat": ac["lat"],
            "lon": ac["lon"],
            "dist_to_route_km": ac["dist_to_route_km"],
            "seen_pos_s": ac["seen_pos_s"],
        })
        print("-" * 80)

    print("\nDone.")


if __name__ == "__main__":
    main()
