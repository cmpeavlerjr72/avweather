"""
test_adsb_lol.py

Free ADS-B feed via adsb.lol
- Query aircraft within a radius around a lat/lon
- Print what fields we actually get

Docs/Examples vary by backend, but this is commonly supported:
https://api.adsb.lol/v2/lat/{lat}/lon/{lon}/dist/{km}
"""

import os
import time
import math
import requests
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

BASE = os.getenv("ADSBLOL_BASE", "https://api.adsb.lol").strip().rstrip("/")

def fetch_near(lat: float, lon: float, dist_km: float) -> dict:
    url = f"{BASE}/v2/lat/{lat}/lon/{lon}/dist/{dist_km}"
    resp = requests.get(url, timeout=30, headers={"User-Agent": "AvWeather-ADSBLOL-Test/1.0"})
    resp.raise_for_status()
    return resp.json()

def nm_to_km(nm: float) -> float:
    return nm * 1.852

def pick_callsign(obj: dict) -> str | None:
    for k in ("flight", "call", "callsign", "cs", "ident"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def normalize(obj: dict) -> dict:
    # Many feeds are readsb/dump1090 style.
    # Common keys: hex, flight, lat, lon, alt_baro, alt_geom, gs, track, baro_rate, type, r(reg), squawk
    alt_baro = obj.get("alt_baro", obj.get("alt"))
    if isinstance(alt_baro, str) and alt_baro.lower() == "ground":
        alt_baro = None

    return {
        "hex": (obj.get("hex") or obj.get("icao") or obj.get("icao24")),
        "callsign": pick_callsign(obj),
        "lat": obj.get("lat"),
        "lon": obj.get("lon"),
        "alt_baro_ft": alt_baro,
        "alt_geom_ft": obj.get("alt_geom", obj.get("galt")),
        "gs_kt": obj.get("gs", obj.get("spd")),
        "track_deg": obj.get("track", obj.get("trak")),
        "vr_fpm": obj.get("baro_rate", obj.get("vsi")),
        "type": obj.get("t", obj.get("type")),
        "reg": obj.get("r", obj.get("reg")),
        "squawk": obj.get("squawk", obj.get("sqk")),
        "seen_pos_s": obj.get("seen_pos"),
        "seen_s": obj.get("seen"),
    }

def main():
    # Defaults: Denver Intl (KDEN)
    center_lat = float(os.getenv("ADSB_CENTER_LAT", "39.8561"))
    center_lon = float(os.getenv("ADSB_CENTER_LON", "-104.6737"))
    radius_nm = float(os.getenv("ADSB_RADIUS_NM", "120"))
    sample_n = int(os.getenv("ADSB_SAMPLE_N", "10"))

    dist_km = nm_to_km(radius_nm)

    print("=== adsb.lol Test ===")
    print(f"Base: {BASE}")
    print(f"Center: ({center_lat}, {center_lon}) radius_nm={radius_nm} (dist_km={dist_km:.1f})")

    t0 = time.time()
    data = fetch_near(center_lat, center_lon, dist_km)
    dt = time.time() - t0

    print("\nTop-level keys:", list(data.keys()))

    # different backends call the list different things
    raw_list = (
        data.get("ac")
        or data.get("aircraft")
        or data.get("planes")
        or data.get("states")
        or []
    )

    print(f"Aircraft list count: {len(raw_list)}")
    if raw_list:
        print("\nFirst raw aircraft object:")
        pprint(raw_list[0])

    norm = [normalize(a) for a in raw_list]

    with_pos = [a for a in norm if a["lat"] is not None and a["lon"] is not None]
    with_callsign = [a for a in with_pos if a["callsign"]]
    airborne_like = [a for a in with_pos if a["alt_baro_ft"] is not None or a["alt_geom_ft"] is not None]

    print(f"\nFetch seconds: {dt:.2f}")
    print(f"Normalized aircraft: {len(norm)}")
    print(f"  with position: {len(with_pos)}")
    print(f"  with callsign: {len(with_callsign)}")
    print(f"  airborne-ish: {len(airborne_like)}")

    print(f"\nSample {min(sample_n, len(with_callsign))} aircraft (with callsign + position):\n")
    for ac in with_callsign[:sample_n]:
        pprint(ac)
        print("-" * 70)

    print("\nDone.")

if __name__ == "__main__":
    main()
