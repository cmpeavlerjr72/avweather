import argparse
import json
import math
from datetime import datetime
from dateutil import tz

import pandas as pd
import requests
from pyproj import Geod, Transformer
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import transform as shp_transform
from shapely.geometry import mapping as shp_mapping
import folium

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from typing import Optional, Tuple

from folium.plugins import FloatImage  # ok to import even if we don't end up using it
from branca.element import MacroElement, Template


def corridor_bbox(route_coords, corridor_poly) -> Tuple[float, float, float, float]:
    """
    Returns (lat_min, lon_min, lat_max, lon_max) as a simple bbox for API queries.
    Uses the corridor polygon if available; falls back to route samples.
    """
    if corridor_poly is not None and hasattr(corridor_poly, "bounds"):
        lon_min, lat_min, lon_max, lat_max = corridor_poly.bounds
        return (lat_min, lon_min, lat_max, lon_max)

    lats = [lat for lat, lon in route_coords]
    lons = [lon for lat, lon in route_coords]
    return (min(lats), min(lons), max(lats), max(lons))

def _json_or_none(resp):
    if resp.status_code == 204:
        return None
    resp.raise_for_status()
    try:
        j = resp.json()
        return j
    except Exception:
        return None

def make_http():
    s = requests.Session()
    s.headers.update({
        # use your real email/URL to avoid being rate-limited
        "User-Agent": "AvWeatherProto/0.2 (+https://your-site.example; contact: your@email)"
    })
    retry = Retry(
        total=4,
        backoff_factor=0.3,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD")
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

HTTP = make_http()


# --- Config ---
GEOD = Geod(ellps="WGS84")
WEBMERCATOR = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True).transform
WGS84 = Transformer.from_crs("epsg:3857", "epsg:4326", always_xy=True).transform

AIRPORT_CSV = "data/airports.csv"

# NOTE: These aviationweather.gov endpoints work widely in practice.
# If you hit CORS/rate limits, add lightweight retry or run from a terminal (not inside a hosted notebook).
# AWC_METAR = "https://aviationweather.gov/data/api/metar"
# AWC_TAF = "https://aviationweather.gov/data/api/taf"

def load_airports():
    df = pd.read_csv(AIRPORT_CSV)
    lut = {r["icao"].strip().upper(): (float(r["lat"]), float(r["lon"]), r["name"]) for _, r in df.iterrows()}
    return lut

def sample_great_circle(lat1, lon1, lat2, lon2, step_nm=50.0):
    """
    Sample points along a great-circle between (lat1,lon1) and (lat2,lon2).
    step_nm: spacing in nautical miles between samples.
    """
    # Compute total great-circle distance (meters) and azimuths
    az12, az21, dist_m = GEOD.inv(lon1, lat1, lon2, lat2)
    total_nm = dist_m / 1852.0
    n_steps = max(2, int(math.ceil(total_nm / step_nm)) + 1)
    lats = []
    lons = []
    for f in [i / (n_steps - 1) for i in range(n_steps)]:
        # interpolate along geodesic
        lon, lat, _ = GEOD.fwd(lon1, lat1, az12, dist_m * f)
        lats.append(lat)
        lons.append(lon)
    return list(zip(lats, lons))

def buffer_route(coords_wgs84, buffer_nm=100.0):
    """
    Buffer a route polyline by buffer_nm (nautical miles) using WebMercator as a fast approximation.
    """
    line = LineString([(lon, lat) for lat, lon in coords_wgs84])  # shapely expects (x=lon, y=lat)
    # project to meters
    line_m = shp_transform(WEBMERCATOR, line)
    buf_m = buffer_nm * 1852.0
    poly_m = line_m.buffer(buf_m, cap_style=2, join_style=2)
    poly_wgs84 = shp_transform(WGS84, poly_m)
    return poly_wgs84  # shapely Polygon in lon/lat


AWC_BASE = "https://aviationweather.gov/api/data"

def fetch_pireps_bbox(lat_min, lon_min, lat_max, lon_max, age_hours: float = 3.0,
                      level: Optional[int] = None, inten: Optional[str] = None, fmt="json"):
    """
    /api/data/pirep with bbox + optional filters.
    fmt: raw | decoded | json | geojson | xml  (we'll use json)
    """
    params = {
        "bbox": f"{lat_min},{lon_min},{lat_max},{lon_max}",
        "format": fmt,
        "age": age_hours
    }
    if level is not None:
        params["level"] = level
    if inten:
        params["inten"] = inten  # lgt|mod|sev (minimum intensity)
    r = HTTP.get(f"{AWC_BASE}/pirep", params=params, timeout=20)
    data = _json_or_none(r)
    if not data: return []
    # The PIREP endpoint returns a list when format=json
    return data if isinstance(data, list) else []

def fetch_gairmet(product="tango", hazard: Optional[str] = None, fore: Optional[int] = None, fmt="geojson"):
    """
    /api/data/gairmet for CONUS G-AIRMETs.
    product: sierra|tango|zulu
    hazard: turb-hi|turb-lo|llws|sfc_wind|ifr|mtn_obs|ice|fzlvl
    fore: 0|3|6|9|12 (forecast hour)
    """
    params = {"product": product, "format": fmt}
    if hazard: params["hazard"] = hazard
    if fore is not None: params["fore"] = fore
    r = HTTP.get(f"{AWC_BASE}/gairmet", params=params, timeout=20)
    j = _json_or_none(r)
    # geojson should give a dict; if not, return empty
    return j if isinstance(j, dict) else None

def fetch_airsigmet(hazard: Optional[str] = None, level: Optional[int] = None, fmt="geojson"):
    """
    /api/data/airsigmet (Domestic SIGMETs for CONUS; convective & non-convective)
    hazard: conv|turb|ice|ifr
    """
    params = {"format": fmt}
    if hazard: params["hazard"] = hazard
    if level is not None: params["level"] = level
    r = HTTP.get(f"{AWC_BASE}/airsigmet", params=params, timeout=20)
    j = _json_or_none(r)
    return j if isinstance(j, dict) else None

def fetch_isigmet(hazard: Optional[str] = None, level: Optional[int] = None, fmt="geojson"):
    """
    /api/data/isigmet (International SIGMETs; does NOT include US domestic format)
    hazard: turb|ice (per docs)
    """
    params = {"format": fmt}
    if hazard: params["hazard"] = hazard
    if level is not None: params["level"] = level
    r = HTTP.get(f"{AWC_BASE}/isigmet", params=params, timeout=20)
    j = _json_or_none(r)
    return j if isinstance(j, dict) else None

def fetch_metars_bbox(lat_min, lon_min, lat_max, lon_max, hours: float = 3.0, fmt="json"):
    params = {
        "bbox": f"{lat_min},{lon_min},{lat_max},{lon_max}",
        "format": fmt,      # raw | decoded | json | geojson | xml | iwxxm
        "hours": hours
    }
    r = HTTP.get(f"{AWC_BASE}/metar", params=params, timeout=20)
    data = _json_or_none(r)
    if not data or not isinstance(data, list):
        return []
    return data

def add_enroute_metars_layer(m: folium.Map, metars: list, corridor_poly, max_labels=40, name="En-route METARs"):
    if not metars:
        return
    fg = folium.FeatureGroup(name=name, show=True)
    count = 0
    for row in metars:
        lat = row.get("lat"); lon = row.get("lon")
        if lat is None or lon is None:
            continue
        # keep only stations inside (or near) the corridor polygon if available
        inside = True
        if corridor_poly is not None and hasattr(corridor_poly, "contains"):
            from shapely.geometry import Point as ShPoint
            try:
                inside = corridor_poly.buffer(0.1).contains(ShPoint(lon, lat))  # small fudge
            except Exception:
                inside = True
        if not inside:
            continue

        raw = row.get("rawOb") or row.get("raw") or row.get("raw_text") or ""
        icao = row.get("icaoId") or ""
        name = row.get("name") or ""
        fltcat = row.get("fltCat") or row.get("flightCategory") or ""
        popup = folium.Popup(
            f"<b>{icao}</b> — {name}<br><b>Flight Cat:</b> {fltcat}<br>"
            f"<pre style='white-space:pre-wrap'>{raw}</pre>",
            max_width=420
        )
        # subtle smaller dot than PIREPs
        folium.CircleMarker(
            [lat, lon], radius=4, color="#4472c4", fill=True, fill_opacity=0.7,
            popup=popup, pane="points_top"
        ).add_to(fg)

        count += 1
        if count >= max_labels:
            break
    fg.add_to(m)


def fetch_metar(icao: str, hours: float = 3.0):
    """
    AWC /api/data/metar — supports format=json.
    Handles 204 (no content) cleanly and tolerates field name variations.
    """
    params = {
        "ids": icao,
        "format": "json",      # per docs: raw | decoded | json | geojson | xml | iwxxm
        "hours": hours,        # default window; adjust as needed
    }
    try:
        r = HTTP.get(f"{AWC_BASE}/metar", params=params, timeout=20)
        if r.status_code == 204:   # valid but no data
            return None
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or not data:
            return None
        row = data[0]
        # Field names per new schema example
        # raw string may appear as "rawOb" (decoded JSON) or "raw" (older), or ADDS "raw_text".
        raw = row.get("rawOb") or row.get("raw") or row.get("raw_text") or ""
        obs_time = row.get("reportTime") or row.get("obsTime") or row.get("observation_time") or ""
        return {"raw": raw, "obsTime": obs_time}
    except Exception as e:
        print(f"[warn] METAR fetch failed for {icao}: {e}")
        return None

def fetch_taf(icao: str):
    """
    AWC /api/data/taf — supports format=json.
    """
    params = {
        "ids": icao,
        "format": "json",
        # you can add ?time=issue or date=... if you want; default is valid time
    }
    try:
        r = HTTP.get(f"{AWC_BASE}/taf", params=params, timeout=20)
        if r.status_code == 204:
            return None
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or not data:
            return None
        row = data[0]
        raw = row.get("rawTAF") or row.get("raw") or row.get("raw_text") or ""
        issue = row.get("issueTime") or row.get("dbPopTime") or ""
        return {"raw": raw, "issueTime": issue}
    except Exception as e:
        print(f"[warn] TAF fetch failed for {icao}: {e}")
        return None

def summarize_corridor(pireps: list, sigmet_gj: dict, gairmet_gjs: list):
    tb_counts = {"LGT":0, "MOD":0, "SEV":0}
    ice_counts = {"LGT":0, "MOD":0, "SEV":0}
    fl_min = None; fl_max = None

    for p in pireps or []:
        fl = p.get("fltLvl")
        try:
            if fl:
                fli = int(str(fl).replace("FL","").strip())
                fl_min = fli if fl_min is None else min(fl_min, fli)
                fl_max = fli if fl_max is None else max(fl_max, fli)
        except Exception:
            pass
        for key, bucket in (("tbInt1", tb_counts), ("tbInt2", tb_counts), ("icgInt1", ice_counts), ("icgInt2", ice_counts)):
            v = (p.get(key) or "").upper()
            for k in ("SEV","MOD","LGT"):
                if k in v:
                    bucket[k]+=1; break

    def _gj_count(gj): 
        try:
            return len((gj or {}).get("features", []))
        except Exception:
            return 0

    sig_cnt = _gj_count(sigmet_gj)
    gar_cnt = sum(_gj_count(g) for g in (gairmet_gjs or []))

    return {
        "tb": tb_counts, "ice": ice_counts,
        "fl_range": (fl_min, fl_max) if fl_min is not None else None,
        "sigmet_count": sig_cnt, "gairmet_count": gar_cnt
    }

def add_summary_box(m: folium.Map, summary: dict, origin: str, dest: str):
    tb = summary["tb"]; ice = summary["ice"]
    fr = summary["fl_range"]
    sigc = summary["sigmet_count"]; gac = summary["gairmet_count"]

    html = f"""
    <div style="
        position: fixed;
        bottom: 14px; left: 14px;
        z-index: 10000;
        background: rgba(255,255,255,0.95);
        padding: 10px 12px;
        border-radius: 10px;
        box-shadow: 0 4px 16px rgba(0,0,0,.25);
        font: 12px/1.3 system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
        max-width: 320px;">
      <div style="font-weight:600; margin-bottom:6px;">Route Summary: {origin} → {dest}</div>
      <div><b>PIREPs (Turb):</b> LGT {tb['LGT']} · MOD {tb['MOD']} · SEV {tb['SEV']}</div>
      <div><b>PIREPs (Icing):</b> LGT {ice['LGT']} · MOD {ice['MOD']} · SEV {ice['SEV']}</div>
      <div><b>Flight Levels:</b> {('FL'+str(fr[0])+'–FL'+str(fr[1])) if fr else 'n/a'}</div>
      <div><b>Active Advisories:</b> SIGMET {sigc} · G-AIRMET {gac}</div>
      <div style="margin-top:6px; color:#666;">Tip: toggle layers (top-right); click PIREP/METAR dots for details.</div>
    </div>
    """
    tpl = Template(f"""{{% macro html(this, kwargs) %}}{html}{{% endmacro %}}""")
    macro = MacroElement(); macro._template = tpl
    m.get_root().add_child(macro)


def make_map(route_coords, corridor_poly, origin, dest, o_name, d_name, o_metar, d_metar, o_taf, d_taf, out_html="route_map.html"):
    # Center map roughly at midpoint
    mid_idx = len(route_coords)//2
    mid_lat, mid_lon = route_coords[mid_idx]

    m = folium.Map(location=[mid_lat, mid_lon], zoom_start=5, tiles="OpenStreetMap")

    # ensure point layers (PIREPs, enroute METARs) are clickable above polygons
    folium.map.CustomPane("points_top", z_index=650).add_to(m)
    folium.map.CustomPane("polys_mid", z_index=620).add_to(m)
    # Route line
    folium.PolyLine(route_coords, weight=4, opacity=0.9, tooltip=f"{origin} → {dest}").add_to(m)

    # Disable pointer events on polygon pane so clicks pass through to markers
    css = """
    <style>
    #polys_mid { pointer-events: none; }  /* let marker clicks pass through */
    </style>
    """
    m.get_root().header.add_child(folium.Element(css))

    # Corridor polygon
    try:
        if corridor_poly is not None and hasattr(corridor_poly, "is_empty") and not corridor_poly.is_empty:
            folium.GeoJson(
                data=shp_mapping(corridor_poly),  # GeoJSON-like dict
                style_function=lambda x: {"fillColor": None, "color": "#3388ff", "weight": 1, "dashArray": "6,4"},
                name="Route Corridor"
            ).add_to(m)
    except Exception as e:
        print(f"[warn] corridor render skipped: {e}")

    # Airport markers with METAR/TAF snippets
    def _mk_popup(icao, name, metar, taf):
        lines = [f"<b>{icao}</b> — {name}"]
        if metar:
            txt = metar.get("raw", metar.get("metar", ""))
            lines.append(f"<pre style='white-space:pre-wrap'>{txt}</pre>")
        if taf:
            txt = taf.get("raw", taf.get("taf", ""))
            if txt:
                lines.append("<b>TAF</b>")
                lines.append(f"<pre style='white-space:pre-wrap'>{txt}</pre>")
        return "<br>".join(lines)

    o_lat, o_lon = route_coords[0]
    d_lat, d_lon = route_coords[-1]

    folium.Marker([o_lat, o_lon],
                  tooltip=f"{origin} (Origin)",
                  popup=folium.Popup(_mk_popup(origin, o_name, o_metar, o_taf), max_width=450)).add_to(m)
    folium.Marker([d_lat, d_lon],
                  tooltip=f"{dest} (Destination)",
                  popup=folium.Popup(_mk_popup(dest, d_name, d_metar, d_taf), max_width=450)).add_to(m)

        # ---- Route corridor bbox → overlay pull/plot ----
    try:
        lat_min, lon_min, lat_max, lon_max = corridor_bbox(route_coords, corridor_poly)
        metars_enroute = fetch_metars_bbox(lat_min, lon_min, lat_max, lon_max, hours=3.0)
        add_enroute_metars_layer(m, metars_enroute, corridor_poly, max_labels=60, name="En-route METARs (≤3h)")

        # PIREPs within bbox and last 3h
        pireps = fetch_pireps_bbox(lat_min, lon_min, lat_max, lon_max, age_hours=3.0)
        add_pireps_layer(m, pireps, name="PIREPs (≤3h)")

        # G-AIRMETs (tango = turbulence; try multiple layers)
        gj_turb = fetch_gairmet(product="tango", hazard=None, fore=None, fmt="geojson")
        add_geojson_polys(m, gj_turb, name="G-AIRMET (Turb)")

        # Sierra (IFR/mtn obs), Zulu (Icing)
        gj_sierra = fetch_gairmet(product="sierra", hazard=None, fore=None, fmt="geojson")
        add_geojson_polys(m, gj_sierra, name="G-AIRMET (IFR/Mtn)")

        gj_zulu = fetch_gairmet(product="zulu", hazard=None, fore=None, fmt="geojson")
        add_geojson_polys(m, gj_zulu, name="G-AIRMET (Icing)")

        # Domestic SIGMETs (convective/non-convective). You can split by hazard if you want:
        sig_all = fetch_airsigmet(hazard=None, level=None, fmt="geojson")
        add_geojson_polys(m, sig_all, name="SIGMET (Domestic)")

        # International SIGMETs (turb/ice only, per docs)
        isig = fetch_isigmet(hazard=None, level=None, fmt="geojson")
        add_geojson_polys(m, isig, name="SIGMET (International)")

        summary = summarize_corridor(pireps, sig_all, [gj_turb, gj_sierra, gj_zulu])
        add_summary_box(m, summary, origin, dest)

    except Exception as e:
        print(f"[warn] overlay fetch/render skipped: {e}")


    folium.LayerControl().add_to(m)
    m.save(out_html)
    print(f"[ok] Wrote {out_html}")

def add_pireps_layer(m: folium.Map, pireps: list, name="PIREPs"):
    """
    Adds PIREPs as circle markers with simple tooltips.
    Colors by turbulence intensity if present; otherwise neutral.
    """
    if not pireps: 
        return
    fg = folium.FeatureGroup(name=name, show=True)
    for p in pireps:
        lat = p.get("lat"); lon = p.get("lon")
        if lat is None or lon is None: 
            continue

        # derive a simple intensity/color
        tb = p.get("tbInt1") or p.get("tbInt2") or ""
        ic = p.get("icgInt1") or p.get("icgInt2") or ""
        raw = p.get("rawOb") or ""
        fl = p.get("fltLvl") or ""
        when = p.get("obsTime") or p.get("receiptTime") or ""

        color = "#3388ff"  # default
        intensity = (tb or ic).upper()
        if "SEV" in intensity:
            color = "#d7191c"
        elif "MOD" in intensity:
            color = "#fdae61"
        elif "LGT" in intensity:
            color = "#abdda4"

        popup = folium.Popup(
            f"<b>PIREPs</b><br>"
            f"FL: {fl}<br>"
            f"TB: {tb} | ICE: {ic}<br>"
            f"Time: {when}<br>"
            f"<pre style='white-space:pre-wrap'>{raw}</pre>",
            max_width=400
        )

        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.8,
            popup=popup,
            pane="points_top"
        ).add_to(fg)
    fg.add_to(m)

def _hazard_style(hazard: str):
    """
    Simple style by hazard type for polygon overlays.
    """
    hazard = (hazard or "").lower()
    color = "#3388ff"
    if "conv" in hazard or "ts" in hazard:
        color = "#d7191c"   # convective
    elif "turb" in hazard or "llws" in hazard:
        color = "#fdae61"
    elif "ice" in hazard or "fzlvl" in hazard:
        color = "#74add1"
    elif "ifr" in hazard or "mtn_obs" in hazard:
        color = "#a6a6a6"
    return {"color": color, "weight": 2, "fillOpacity": 0.15}

def add_geojson_polys(m: folium.Map, gj: dict, name: str):
    """
    Generic GeoJSON polygon/line overlay with hazard-aware styling if property exists.
    """
    if not gj: 
        return
    def styler(feat):
        hz = None
        props = feat.get("properties", {}) if isinstance(feat, dict) else {}
        # common property names that carry hazard/type
        hz = props.get("hazard") or props.get("type") or props.get("phenom") or ""
        return _hazard_style(hz)
    folium.GeoJson(gj, name=name, style_function=styler, pane="polys_mid").add_to(m)


def main():
    ap = argparse.ArgumentParser(description="Route corridor + endpoint METAR/TAF demo")
    ap.add_argument("origin", help="Origin ICAO (e.g., KATL)")
    ap.add_argument("destination", help="Destination ICAO (e.g., KDEN)")
    ap.add_argument("--step-nm", type=float, default=50, help="Great-circle sample spacing (NM)")
    ap.add_argument("--buffer-nm", type=float, default=100, help="Route corridor half-width (NM)")
    ap.add_argument("--out", default="route_map.html", help="Output HTML map filename")
    args = ap.parse_args()

    airports = load_airports()
    o = args.origin.strip().upper()
    d = args.destination.strip().upper()
    if o not in airports or d not in airports:
        raise SystemExit(f"Missing ICAO in airports.csv — add {o} or {d}")

    o_lat, o_lon, o_name = airports[o]
    d_lat, d_lon, d_name = airports[d]

    # Great-circle samples
    coords = sample_great_circle(o_lat, o_lon, d_lat, d_lon, step_nm=args.step_nm)

    # Buffer/corridor polygon (for later weather overlays)
    corridor = buffer_route(coords, buffer_nm=args.buffer_nm)

    # Quick weather pulls (Origin/Dest only for now)
    o_metar = fetch_metar(o)
    d_metar = fetch_metar(d)
    o_taf = fetch_taf(o)
    d_taf = fetch_taf(d)

    make_map(coords, corridor, o, d, o_name, d_name, o_metar, d_metar, o_taf, d_taf, out_html=args.out)

if __name__ == "__main__":
    main()
