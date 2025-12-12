import os
import time
import math
import requests
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

ADSBX_API_AUTH = (os.getenv("ADSBX_API_AUTH") or "").strip()
REST_BASE = "https://adsbexchange.com/api/aircraft"
GLOBE_SNAPSHOT_URL = "https://globe.adsbexchange.com/data/aircraft.json"

def _nm_to_km(nm: float) -> float:
    return nm * 1.852

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def _pick_callsign(obj: dict) -> str | None:
    for k in ("flight", "call", "callsign", "cs", "ident"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def fetch_rest_near_point(lat: float, lon: float, dist_nm: int = 100) -> dict:
    if not ADSBX_API_AUTH:
        raise RuntimeError("Missing ADSBX_API_AUTH in .env (needed for REST API mode).")
    url = f"{REST_BASE}/lat/{lat}/lon/{lon}/dist/{dist_nm}/"
    resp = requests.get(
        url,
        timeout=30,
        headers={"api-auth": ADSBX_API_AUTH, "User-Agent": "AvWeather-ADSBX-Test/1.1"},
    )
    resp.raise_for_status()
    return resp.json()

def fetch_globe_snapshot() -> dict:
    resp = requests.get(GLOBE_SNAPSHOT_URL, timeout=60, headers={"User-Agent": "AvWeather-ADSBX-Test/1.1"})
    resp.raise_for_status()
    return resp.json()

def normalize(obj: dict) -> dict:
    alt_baro = obj.get("alt_baro")
    if isinstance(alt_baro, str) and alt_baro.lower() == "ground":
        alt_baro = None

    return {
        "hex": obj.get("hex") or obj.get("icao") or obj.get("icao24"),
        "callsign": _pick_callsign(obj),
        "lat": obj.get("lat"),
        "lon": obj.get("lon"),
        "alt_baro_ft": alt_baro,
        "alt_geom_ft": obj.get("alt_geom") or obj.get("geom_alt"),
        "gs_kt": obj.get("gs") or obj.get("spd"),
        "track_deg": obj.get("track") or obj.get("trk"),
        "vr_fpm": obj.get("baro_rate") or obj.get("vrt"),
        "type": obj.get("t") or obj.get("type"),
        "reg": obj.get("r") or obj.get("reg"),
        "squawk": obj.get("squawk"),
        "seen_pos_s": obj.get("seen_pos"),
        "seen_s": obj.get("seen"),
    }

def filter_by_radius(ac_list: list[dict], center_lat: float, center_lon: float, radius_nm: float) -> list[dict]:
    r_km = _nm_to_km(radius_nm)
    out = []
    for ac in ac_list:
        lat, lon = ac.get("lat"), ac.get("lon")
        if lat is None or lon is None:
            continue
        if _haversine_km(center_lat, center_lon, lat, lon) <= r_km:
            out.append(ac)
    return out

def main():
    MODE = os.getenv("ADSBX_MODE", "rest").strip().lower()
    CENTER_LAT = float(os.getenv("ADSBX_CENTER_LAT", "39.8561"))
    CENTER_LON = float(os.getenv("ADSBX_CENTER_LON", "-104.6737"))
    RADIUS_NM = float(os.getenv("ADSBX_RADIUS_NM", "120"))
    SAMPLE_N = int(os.getenv("ADSBX_SAMPLE_N", "10"))

    print("=== ADS-B Exchange Test (debug) ===")
    print(f"Mode: {MODE}")
    print(f"Center: ({CENTER_LAT}, {CENTER_LON}), radius_nm={RADIUS_NM}")
    print(f"Have ADSBX_API_AUTH: {bool(ADSBX_API_AUTH)}")

    t0 = time.time()

    if MODE == "rest":
        dist_nm = min(int(round(RADIUS_NM)), 100)
        print(f"\nFetching via REST API dist={dist_nm}nm...")
        data = fetch_rest_near_point(CENTER_LAT, CENTER_LON, dist_nm=dist_nm)

        print("\nTop-level response keys:", list(data.keys()))
        # Common keys: 'ac' or 'aircraft'
        raw_list = data.get("ac")
        used_key = "ac"
        if raw_list is None:
            raw_list = data.get("aircraft")
            used_key = "aircraft"

        raw_list = raw_list or []
        print(f"Aircraft list key used: {used_key} (count={len(raw_list)})")

        if raw_list:
            print("\nFirst raw aircraft object (as returned):")
            pprint(raw_list[0])

        norm = [normalize(a) for a in raw_list]

    else:
        print("\nFetching public globe snapshot aircraft.json...")
        data = fetch_globe_snapshot()
        raw_list = data.get("aircraft") or []
        print("Top-level response keys:", list(data.keys()))
        print(f"Snapshot aircraft count: {len(raw_list)}")
        if raw_list:
            print("\nFirst raw aircraft object (snapshot):")
            pprint(raw_list[0])

        norm = [normalize(a) for a in raw_list]
        norm = filter_by_radius(norm, CENTER_LAT, CENTER_LON, RADIUS_NM)

    dt = time.time() - t0

    with_pos = [a for a in norm if a["lat"] is not None and a["lon"] is not None]
    with_callsign = [a for a in with_pos if a["callsign"]]
    airborne_like = [a for a in with_pos if a["alt_baro_ft"] is not None or a["alt_geom_ft"] is not None]

    print(f"\nFetch+parse seconds: {dt:.2f}")
    print(f"Aircraft in working set: {len(norm)}")
    print(f"  with position: {len(with_pos)}")
    print(f"  with callsign: {len(with_callsign)}")
    print(f"  airborne-ish: {len(airborne_like)}")

    print(f"\nSample {min(SAMPLE_N, len(with_pos))} normalized aircraft (with position):\n")
    for ac in with_pos[:SAMPLE_N]:
        pprint(ac)
        print("-" * 70)

    print("\nDone.")

if __name__ == "__main__":
    main()
