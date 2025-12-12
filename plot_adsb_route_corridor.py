"""
plot_adsb_route_corridor_map.py

Creates an interactive Leaflet map (folium) showing:
- Route line (ORIG -> DEST)
- Sample query points
- Aircraft positions found in the corridor
- A short "track" line in the heading direction for each aircraft

Output: adsb_route_map.html (open in browser)
"""

import os
import math
import time
import requests
import folium

BASE = os.getenv("ADSBLOL_BASE", "https://api.adsb.lol").strip().rstrip("/")

# Default route: KMSP -> KDEN (edit as needed)
ORIG = (float(os.getenv("ORIG_LAT", "44.8848")), float(os.getenv("ORIG_LON", "-93.2223")))   # KMSP
DEST = (float(os.getenv("DEST_LAT", "39.8561")), float(os.getenv("DEST_LON", "-104.6737")))  # KDEN

SAMPLES = int(os.getenv("ROUTE_SAMPLES", "7"))
QUERY_RADIUS_NM = float(os.getenv("QUERY_RADIUS_NM", "45"))
MIN_ALT_FT = float(os.getenv("MIN_ALT_FT", "10000"))

# Track visualization (km)
TRACK_LINE_KM = float(os.getenv("TRACK_LINE_KM", "25"))  # how long the heading "ray" should be

# Output
OUT_HTML = os.getenv("ADSB_MAP_OUT", "adsb_route_map.html")


def nm_to_km(nm: float) -> float:
    return nm * 1.852


def interpolate_points(lat1, lon1, lat2, lon2, n: int):
    pts = []
    for i in range(n):
        t = 0 if n == 1 else i / (n - 1)
        lat = lat1 + (lat2 - lat1) * t
        lon = lon1 + (lon2 - lon1) * t
        pts.append((lat, lon))
    return pts


def fetch_near(lat: float, lon: float, dist_km: float) -> dict:
    url = f"{BASE}/v2/lat/{lat}/lon/{lon}/dist/{dist_km}"
    resp = requests.get(url, timeout=30, headers={"User-Agent": "AvWeather-RouteMap/1.0"})
    resp.raise_for_status()
    return resp.json()


def pick_callsign(obj: dict) -> str | None:
    for k in ("flight", "call", "callsign", "cs", "ident"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def normalize(obj: dict) -> dict:
    track = obj.get("track") or obj.get("dir") or obj.get("trak")
    return {
        "hex": obj.get("hex") or obj.get("icao") or obj.get("icao24"),
        "callsign": pick_callsign(obj),
        "lat": obj.get("lat"),
        "lon": obj.get("lon"),
        "alt_baro_ft": obj.get("alt_baro", obj.get("alt")),
        "alt_geom_ft": obj.get("alt_geom", obj.get("galt")),
        "gs_kt": obj.get("gs", obj.get("spd")),
        "track_deg": track,
        "vr_fpm": obj.get("baro_rate", obj.get("vsi")),
        "reg": obj.get("r", obj.get("reg")),
        "type": obj.get("t", obj.get("type")),
        "squawk": obj.get("squawk", obj.get("sqk")),
        "seen_pos_s": obj.get("seen_pos"),
        "seen_s": obj.get("seen"),
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


def destination_point(lat: float, lon: float, bearing_deg: float, distance_km: float):
    """
    Great-circle destination point given start, bearing, and distance.
    Bearing: degrees, where 0 = north, 90 = east.
    """
    R = 6371.0
    brng = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    d = distance_km / R

    lat2 = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng) * math.sin(d) * math.cos(lat1),
                             math.cos(d) - math.sin(lat1) * math.sin(lat2))

    return (math.degrees(lat2), (math.degrees(lon2) + 540) % 360 - 180)  # normalize lon


def main():
    print("Running corridor query...")
    route_samples = interpolate_points(ORIG[0], ORIG[1], DEST[0], DEST[1], SAMPLES)
    dist_km = nm_to_km(QUERY_RADIUS_NM)

    all_raw = []
    t0 = time.time()
    for idx, (lat, lon) in enumerate(route_samples, 1):
        print(f"Query {idx}/{len(route_samples)} at ({lat:.4f}, {lon:.4f})")
        payload = fetch_near(lat, lon, dist_km)
        raw_list = payload.get("ac") or payload.get("aircraft") or payload.get("planes") or []
        all_raw.extend(raw_list)
        time.sleep(0.15)
    print(f"Fetched raw rows: {len(all_raw)} in {time.time() - t0:.2f}s")

    # Normalize + dedupe
    by_hex = {}
    for obj in all_raw:
        ac = normalize(obj)
        hx = ac.get("hex")
        if not hx:
            continue
        if ac["lat"] is None or ac["lon"] is None:
            continue
        if alt_ft(ac) < MIN_ALT_FT:
            continue

        prev = by_hex.get(hx)
        if prev is None:
            by_hex[hx] = ac
        else:
            prev_seen = prev.get("seen_pos_s")
            cur_seen = ac.get("seen_pos_s")
            if prev_seen is None or (cur_seen is not None and cur_seen < prev_seen):
                by_hex[hx] = ac

    aircraft = list(by_hex.values())
    print(f"After filters: {len(aircraft)} aircraft")

    # Create map centered mid-route
    mid_lat = (ORIG[0] + DEST[0]) / 2
    mid_lon = (ORIG[1] + DEST[1]) / 2
    m = folium.Map(location=[mid_lat, mid_lon], zoom_start=6, tiles="OpenStreetMap")

    # Route line
    folium.PolyLine(locations=[ORIG, DEST], weight=4, opacity=0.8).add_to(m)

    # Origin/Dest markers
    folium.Marker(location=list(ORIG), tooltip="ORIG", popup=f"ORIG: {ORIG}").add_to(m)
    folium.Marker(location=list(DEST), tooltip="DEST", popup=f"DEST: {DEST}").add_to(m)

    # Sample points
    for i, (lat, lon) in enumerate(route_samples, 1):
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            fill=True,
            fill_opacity=0.9,
            tooltip=f"Sample {i}",
        ).add_to(m)

        # Optional: show each query radius
        folium.Circle(
            location=[lat, lon],
            radius=dist_km * 1000,  # meters
            weight=1,
            opacity=0.15,
            fill=False,
        ).add_to(m)

    # Aircraft markers + track lines
    for ac in aircraft:
        lat, lon = float(ac["lat"]), float(ac["lon"])
        cs = ac.get("callsign") or "(no callsign)"
        aalt = ac.get("alt_baro_ft")
        gs = ac.get("gs_kt")
        trk = ac.get("track_deg")

        popup = (
            f"Callsign: {cs}<br>"
            f"Hex: {ac.get('hex')}<br>"
            f"Type: {ac.get('type')}<br>"
            f"Reg: {ac.get('reg')}<br>"
            f"Alt(ft): {aalt}<br>"
            f"GS(kt): {gs}<br>"
            f"Track(deg): {trk}<br>"
            f"VRate: {ac.get('vr_fpm')}<br>"
            f"Seen_pos(s): {ac.get('seen_pos_s')}"
        )

        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            fill=True,
            fill_opacity=0.9,
            tooltip=cs,
            popup=folium.Popup(popup, max_width=350),
        ).add_to(m)

        # Draw heading ray if we have track
        if trk is not None:
            try:
                end_lat, end_lon = destination_point(lat, lon, float(trk), TRACK_LINE_KM)
                folium.PolyLine(
                    locations=[(lat, lon), (end_lat, end_lon)],
                    weight=2,
                    opacity=0.7,
                ).add_to(m)
            except Exception:
                pass

    m.save(OUT_HTML)
    print(f"\nSaved map to: {OUT_HTML}")
    print("Open it in your browser (double click the file).")


if __name__ == "__main__":
    main()
