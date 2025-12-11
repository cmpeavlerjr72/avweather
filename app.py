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
from shapely.geometry import shape as shp_shape, Point as ShPoint
from folium.plugins import FloatImage  # ok to import even if we don't end up using it
from branca.element import MacroElement, Template
import html as _html
import os, json
from openai import OpenAI


import os
from dotenv import load_dotenv

# Load .env from project root
load_dotenv()  # looks for .env up the tree
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise SystemExit("OPENAI_API_KEY not set. Put it in .env or environment.")
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- GPT wiring copied from the working CBB script (adapted) ----------

# same idea as your basketball file: try chat first, then responses
MODEL_FALLBACKS = ["gpt-5-mini", "gpt-4o-mini", "gpt-5"]

AVIATION_SYSTEM_PROMPT = (
    "Write a concise, passenger-friendly en-route briefing in the voice and clarity of an airline captain, "
    "but do NOT present yourself as the actual pilot or crew for this flight. "
    "Avoid first-person statements that imply operational control (e.g., 'I', 'we', 'this is your captain speaking'). "
    "Use neutral phrasing like 'Passengers can expect…', 'The flight may encounter…', 'Around X minutes in…', "
    "and keep it calm, confident, factual, and time-oriented. Do not invent data."
)

def _as_paragraphs(text: str) -> str:
    """Convert a GPT string with blank lines into <p> paragraphs, HTML-escaped."""
    if not text:
        return ""
    # Escape HTML, then split on double newlines
    safe = _html.escape(text.strip())
    parts = [p.strip() for p in safe.split("\n\n") if p.strip()]
    return "<p>" + "</p><p>".join(parts) + "</p>"


def build_captains_user_prompt(origin: str, dest: str, total_min: float, analysis: dict) -> str:
    """
    Build a compact, purely factual JSON-like text the model can turn into a briefing.
    """
    lines = []
    lines.append(f"Route: {origin} → {dest}")
    lines.append(f"Estimated_time_minutes: {round(total_min,1)}")
    # PIREPs timing/events
    tb_events = analysis.get("tb_events", [])
    ice_events = analysis.get("ice_events", [])
    fl_rng = analysis.get("flight_level_range")
    advisories = analysis.get("advisories", {})
    metar_notes = analysis.get("metar_notes", [])

    if tb_events:
        # list a few timed events: (minute, level)
        chunks = "; ".join(f"{int(t)}min:{lvl}" for t, lvl in tb_events[:12])
        lines.append(f"Turbulence_events: {chunks}")
    else:
        lines.append("Turbulence_events: none reported")

    if ice_events:
        chunks = "; ".join(f"{int(t)}min:{lvl}" for t, lvl in ice_events[:12])
        lines.append(f"Icing_events: {chunks}")
    else:
        lines.append("Icing_events: none reported")

    if fl_rng:
        a, b = fl_rng
        lines.append(f"Reported_flight_levels_range: FL{a}–FL{b}")
    else:
        lines.append("Reported_flight_levels_range: unknown")

    lines.append(f"Advisories: SIGMET={advisories.get('sigmet',0)}, G-AIRMET={advisories.get('gairmet',0)}")

    if metar_notes:
        # show a few stations that indicate precip/IFR along the corridor
        snips = "; ".join(
            f"{m.get('icao','')}: {(m.get('wx') or '').upper()} {(m.get('cat') or '')}".strip()
            for m in metar_notes[:10]
        )
        lines.append(f"Enroute_station_flags: {snips}")
    else:
        lines.append("Enroute_station_flags: none notable")

    # dynamic reassurance guidance (only include what appears in analysis)
    rassure = []
    if tb_events:
        rassure.append("Turbulence is normal in flight; aircraft are built to withstand it and crews adjust speed/altitude to keep it comfortable.")
    if ice_events:
        rassure.append("Modern jets have anti-ice systems and procedures; crews monitor and avoid prolonged icing when needed.")
    if advisories.get("sigmet", 0) > 0 or advisories.get("gairmet", 0) > 0:
        rassure.append("Air traffic control and onboard weather radar help crews reroute around rough weather when beneficial.")

    # formatting + style constraints
    lines.append(
        "Write 2–3 short paragraphs (separated by a blank line), captain-style plain English. "
        "Do NOT imply you are the operating crew. Avoid codes/jargon. "
        "Mention approximate timing like 'about 30 minutes in'. "
        "If applicable, briefly explain why turbulence/icing/advisories are managed safely for passengers."
        + ("" if not rassure else " Reassurance points to consider: " + " ".join(rassure))
    )


    return "\n".join(lines)

def _chat_completions(user_prompt: str, model: str, max_tokens: int):
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": AVIATION_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        "max_completion_tokens": max_tokens,   # <- works with your env
        # no temperature (your model rejected custom temps)
    }
    resp = client.chat.completions.create(**kwargs)
    try:
        raw = resp.to_dict()
    except Exception:
        raw = None
    text = (resp.choices[0].message.content or "").strip()
    return text, raw

def _responses_api(user_prompt: str, model: str, max_tokens: int):
    # mirror your working script’s content shape
    messages = [
        {"role": "system", "content": [{"type": "text", "text": AVIATION_SYSTEM_PROMPT}]},
        {"role": "user",   "content": [{"type": "text", "text": user_prompt}]},
    ]
    r = client.responses.create(
        model=model,
        input=messages,
        max_output_tokens=max_tokens,
    )
    text = (getattr(r, "output_text", None) or "").strip()
    if not text:
        # assemble manually if SDK returns the older shape
        parts = []
        output = getattr(r, "output", None)
        if isinstance(output, list):
            for item in output:
                content = getattr(item, "content", None)
                if isinstance(content, list):
                    for chunk in content:
                        t = chunk.get("text") or chunk.get("value") or ""
                        if isinstance(t, str) and t:
                            parts.append(t)
        text = "\n".join(parts).strip()
    try:
        raw = r.to_dict()
    except Exception:
        raw = None
    return text, raw

def call_model_with_retries_for_briefing(user_prompt: str, primary_model: str, max_tokens: int):
    tried = []

    # 1) Chat completions attempts
    for m in [primary_model] + [x for x in MODEL_FALLBACKS if x != primary_model]:
        try:
            text, _ = _chat_completions(user_prompt, m, max_tokens)
            tried.append((m, "chat"))
            if text:
                return text
        except Exception as e:
            tried.append((m, f"chat_error:{e}"))
            continue

    # 2) Responses API attempts
    for m in [primary_model] + [x for x in MODEL_FALLBACKS if x != primary_model]:
        try:
            text, _ = _responses_api(user_prompt, m, max_tokens)
            tried.append((m, "responses"))
            if text:
                return text
        except Exception as e:
            tried.append((m, f"responses_error:{e}"))
            continue

    print(f"[warn] briefing: all attempts empty. Tried: {tried}")
    return ""



# Optional: improve UA with your .env values
APP_NAME = os.getenv("APP_NAME", "AvWeatherProto")
APP_URL = os.getenv("APP_URL", "https://example.com")
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "you@example.com")


def _gpt_chat_completion(model: str, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        # IMPORTANT for GPT-5 family:
        max_completion_tokens=max_tokens,
        temperature=0.4,
    )
    return (resp.choices[0].message.content or "").strip()

def _gpt_responses(model: str, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    # Some SDKs prefer "input" as structured messages
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user",   "content": [{"type": "text", "text": user_prompt}]},
    ]
    r = client.responses.create(
        model=model,
        input=messages,
        max_output_tokens=max_tokens,
        temperature=0.4,
    )
    # Try the convenience accessor first:
    txt = (getattr(r, "output_text", None) or "").strip()
    if txt:
        return txt
    # Fallback: assemble text from chunks
    try:
        parts = []
        output = getattr(r, "output", None)
        if isinstance(output, list):
            for item in output:
                content = getattr(item, "content", None)
                if isinstance(content, list):
                    for chunk in content:
                        t = chunk.get("text") or chunk.get("value") or ""
                        if isinstance(t, str) and t:
                            parts.append(t)
        return "\n".join(parts).strip()
    except Exception:
        return ""


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
        "User-Agent": f"{APP_NAME}/0.2 (+{APP_URL}; contact: {CONTACT_EMAIL})"
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

def buffer_poly_nm(poly, nm):
    """Buffer a WGS84 polygon by NM using WebMercator meters, return back in WGS84."""
    if poly is None:
        return None
    poly_m = shp_transform(WEBMERCATOR, poly)
    buf_m  = poly_m.buffer(float(nm) * 1852.0)
    return shp_transform(WGS84, buf_m)

def bounds_of(poly):
    """Return (lat_min, lon_min, lat_max, lon_max) for a WGS84 polygon."""
    if poly is None or not hasattr(poly, "bounds"):
        return None
    lon_min, lat_min, lon_max, lat_max = poly.bounds
    return lat_min, lon_min, lat_max, lon_max


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

def add_enroute_metars_layer(m: folium.Map, metars: list, corridor_poly,
                             max_labels=40, name="En-route METARs",
                             only_flags=False, show=True):
    if not metars:
        return
    fg = folium.FeatureGroup(name=name, show=show)
    count = 0
    for row in metars:
        lat = row.get("lat"); lon = row.get("lon")
        if lat is None or lon is None: continue
        inside = True
        if corridor_poly is not None and hasattr(corridor_poly, "contains"):
            from shapely.geometry import Point as ShPoint
            try:
                inside = corridor_poly.buffer(0.1).contains(ShPoint(lon, lat))
            except Exception:
                inside = True
        if not inside: continue

        wx  = (row.get("wxString") or "").upper()
        cat = (row.get("fltCat") or row.get("flightCategory") or "")
        if only_flags:
            if not ("TS" in wx or "RA" in wx or "SN" in wx or "SH" in wx or cat in ("IFR","LIFR")):
                continue

        raw = row.get("rawOb") or row.get("raw") or row.get("raw_text") or ""
        icao = row.get("icaoId") or ""
        name_air = row.get("name") or ""
        fltcat = cat
        popup = folium.Popup(
            f"<b>{icao}</b> — {name_air}<br><b>Flight Cat:</b> {fltcat}<br>"
            f"<pre style='white-space:pre-wrap'>{raw}</pre>",
            max_width=420
        )
        folium.CircleMarker([lat, lon], radius=4, color="#2b6cb0", fill=True, fill_opacity=0.75,
                            popup=popup, pane="points_top").add_to(fg)
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

def _within_fl(p, fl, band=4000):
    fl_str = p.get("fltLvl")
    try:
        rep = int(str(fl_str).replace("FL","").strip())
        return abs(rep - fl) <= band
    except Exception:
        return False


def make_map(
    route_coords,
    corridor_poly,
    origin,
    dest,
    o_name,
    d_name,
    o_metar,
    d_metar,
    o_taf,
    d_taf,
    out_html="route_map.html",
    calm=False,
    cruise_fl=340,
    pirep_age=1.0,
    metar_hours=1.0,
    pirep_corridor_buffer_nm=20.0,  # <- add default to avoid "non-default after default" error
    ):



    # Center map roughly at midpoint
    mid_idx = len(route_coords)//2
    mid_lat, mid_lon = route_coords[mid_idx]

    # Center will be overridden by fit_bounds if we have a corridor
    m = folium.Map(location=[mid_lat, mid_lon],
                zoom_start=5 if not calm else 6,
                tiles="CartoDB positron" if calm else "OpenStreetMap")
    # If we have a corridor, fit to its bounds with a little padding
    try:
        if corridor_poly is not None and hasattr(corridor_poly, "bounds"):
            lon_min, lat_min, lon_max, lat_max = corridor_poly.bounds
            m.fit_bounds([[lat_min, lon_min], [lat_max, lon_max]], padding=(20, 20))
            # keep PIREPs only slightly outside the corridor (tunable)
            corridor_pirep = buffer_poly_nm(corridor_poly, pirep_corridor_buffer_nm)

            # use the buffered polygon's bbox for the API call (tighter than the old bbox)
            b = bounds_of(corridor_pirep) or corridor_bbox(route_coords, corridor_poly)
            lat_min, lon_min, lat_max, lon_max = b

    except Exception:
        pass


    # ensure point layers (PIREPs, enroute METARs) are clickable above polygons
    folium.map.CustomPane("points_top", z_index=650).add_to(m)
    folium.map.CustomPane("polys_mid", z_index=620).add_to(m)
    # Route line
    folium.PolyLine(route_coords,
                weight=5 if calm else 4,
                color="#b02b2b", opacity=0.95, tooltip=f"{origin} → {dest}").add_to(m)

    # Disable pointer events on polygon pane so clicks pass through to markers
    css = """
    <style>
    .leaflet-pane.polys_mid { pointer-events: none; }  /* let marker clicks pass through */
    </style>
    """

    m.get_root().header.add_child(folium.Element(css))

    # Corridor polygon
    try:
        # Corridor polygon faint
        if corridor_poly is not None and hasattr(corridor_poly, "is_empty") and not corridor_poly.is_empty:
            folium.GeoJson(
                data=shp_mapping(corridor_poly),
                style_function=lambda x: {
                    "fillColor": "#2b6cb0" if calm else None,
                    "fillOpacity": 0.06 if calm else 0.0,
                    "color": "#2b6cb0",
                    "opacity": 0.35 if calm else 0.6,
                    "weight": 1,
                    "dashArray": "6,4" if not calm else None
                },
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
        lat_min, lon_min, lat_max, lon_max = (
                bounds_of(corridor_pirep) or corridor_bbox(route_coords, corridor_poly)
            )

        metars_enroute = fetch_metars_bbox(lat_min, lon_min, lat_max, lon_max, hours=metar_hours)
        add_enroute_metars_layer(
                                    m, metars_enroute, corridor_poly,
                                    max_labels=20 if calm else 60,
                                    name="En-route METARs (flags)",
                                    only_flags=calm,
                                    show=True
                                )


        # PIREPs within bbox and last 3h
        pireps = fetch_pireps_bbox(lat_min, lon_min, lat_max, lon_max, age_hours=pirep_age)

        # Filter by cruise FL (±4000 ft) first
        pireps = [p for p in pireps if _within_fl(p, cruise_fl, band=4000)]

        # Now plot ONLY those within the buffered corridor
        add_pireps_layer(
            m, pireps,
            name=f"PIREPs (≤{pirep_age}h, ≥{'MOD' if calm else 'LGT'})",
            show=True,
            min_tb="MOD" if calm else "LGT",
            only_inside=corridor_pirep  # <-- key line
        )




        # G-AIRMETs (tango = turbulence; try multiple layers)
        gj_turb = fetch_gairmet(product="tango", hazard=None, fore=None, fmt="geojson")
        gj_turb  = _intersecting_features(gj_turb, corridor_poly,cruise_fl)
        add_geojson_polys(m, gj_turb, name="G-AIRMET (Turb)",        show=not calm)

        # Sierra (IFR/mtn obs), Zulu (Icing)
        gj_sierra = fetch_gairmet(product="sierra", hazard=None, fore=None, fmt="geojson")
        gj_sierra = _intersecting_features(gj_sierra, corridor_poly,cruise_fl)
        add_geojson_polys(m, gj_sierra, name="G-AIRMET (IFR/Mtn)",        show=not calm)

        gj_zulu = fetch_gairmet(product="zulu", hazard=None, fore=None, fmt="geojson")
        gj_zulu  = _intersecting_features(gj_zulu, corridor_poly,cruise_fl)
        add_geojson_polys(m, gj_zulu, name="G-AIRMET (Icing)",        show=not calm)

        # Domestic SIGMETs (convective/non-convective). You can split by hazard if you want:
        sig_all = fetch_airsigmet(hazard=None, level=None, fmt="geojson")
        sig_all  = _intersecting_features(sig_all, corridor_poly,cruise_fl)
        add_geojson_polys(m, sig_all, name="SIGMET (Domestic)",        show=not calm)

        # International SIGMETs (turb/ice only, per docs)
        isig = fetch_isigmet(hazard=None, level=None, fmt="geojson")
        isig     = _intersecting_features(isig, corridor_poly,cruise_fl)
        add_geojson_polys(m, isig, name="SIGMET (International)",        show=not calm)

        summary = summarize_corridor(pireps, sig_all, [gj_turb, gj_sierra, gj_zulu])
        add_summary_box(m, summary, origin, dest)

                # --- Narrative from GPT ---
        try:
            # total route time estimate (assuming ~450 kt)
            _, total_m = _meters_route_line(route_coords)
            total_nm  = total_m / 1852.0
            total_min = (total_nm / 450.0) * 60.0

            analysis = analyze_enroute(
                route_coords,
                pireps,
                metars_enroute,
                sig_all,
                [gj_turb, gj_sierra, gj_zulu]
            )
            briefing = gpt_summary(origin, dest, analysis, total_min, model="gpt-5")
            if not briefing:
                briefing = "Briefing unavailable right now. Layers remain live—click PIREP/METAR dots for details."

            briefing_html = _as_paragraphs(briefing)

            html = f"""
            <div style="
                position: fixed; top: 14px; left: 14px; z-index: 10000;
                background: rgba(255,255,255,0.97); padding: 12px 14px; border-radius: 10px;
                box-shadow: 0 4px 16px rgba(0,0,0,.25);
                font: 13px/1.45 system-ui,-apple-system,'Segoe UI',Roboto,Arial;">
            <div style="font-weight:600; margin-bottom:6px;">Captain-style Briefing (not the operating crew)</div>
            <div style="max-width: 460px; max-height: 260px; overflow:auto;">{briefing_html}</div>

            </div>
            """

            print(f"[info] Briefing len: {len(briefing)} chars")

            tpl = Template(f"""{{% macro html(this, kwargs) %}}{html}{{% endmacro %}}""")
            macro = MacroElement(); macro._template = tpl
            m.get_root().add_child(macro)
        except Exception as e:
            print(f"[warn] GPT briefing skipped: {e}")


    except Exception as e:
        print(f"[warn] overlay fetch/render skipped: {e}")


    folium.LayerControl().add_to(m)
    m.save(out_html)
    print(f"[ok] Wrote {out_html}")

def add_pireps_layer(m: folium.Map, pireps: list, name="PIREPs",
                     show=True, min_tb="MOD", only_inside=None):
    """
    Adds PIREPs as circle markers.
    min_tb: 'LGT'|'MOD'|'SEV'  (filter by minimum TB/ICE intensity found on the report)
    only_inside: shapely Polygon to keep only points inside (buffered slightly)
    """
    if not pireps:
        return
    fg = folium.FeatureGroup(name=name, show=show)
    rank = {"LGT": 1, "MOD": 2, "SEV": 3}

    for p in pireps:
        lat = p.get("lat"); lon = p.get("lon")
        if lat is None or lon is None: 
            continue

        # Corridor filter
        if only_inside is not None:
            try:
                from shapely.geometry import Point as ShPoint
                if not only_inside.buffer(0.1).contains(ShPoint(lon, lat)):
                    continue
            except Exception:
                pass

        tb = (p.get("tbInt1") or p.get("tbInt2") or "").upper()
        ic = (p.get("icgInt1") or p.get("icgInt2") or "").upper()
        # Decide the strongest mentioned
        def strongest(v):
            if "SEV" in v: return "SEV"
            if "MOD" in v: return "MOD"
            if "LGT" in v: return "LGT"
            return None
        intensity = strongest(tb) or strongest(ic)
        if intensity and rank[intensity] < rank[min_tb]:
            continue  # below threshold

        color = {"SEV": "#c0392b", "MOD": "#e67e22", "LGT": "#7fbf7b"}.get(intensity, "#3388ff")
        raw = p.get("rawOb") or ""
        fl = p.get("fltLvl") or ""
        when = p.get("obsTime") or p.get("receiptTime") or ""

        popup = folium.Popup(
            f"<b>PIREPs</b><br>FL: {fl}<br>TB: {tb} | ICE: {ic}<br>Time: {when}<br>"
            f"<pre style='white-space:pre-wrap'>{raw}</pre>",
            max_width=400
        )
        folium.CircleMarker([lat, lon], radius=5, color=color, fill=True, fill_opacity=0.85,
                            popup=popup, pane="points_top").add_to(fg)
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

def _to_fl(v):
    """Normalize alt to flight levels. Accepts feet or FL."""
    if v is None:
        return None
    try:
        v = int(str(v).replace("FL", "").strip())
    except Exception:
        return None
    # crude: if it's > 500, assume feet → FL
    return v // 100 if v > 500 else v

def _poly_overlaps_alt(props: dict, cruise_fl: int, pad_fl: int = 20) -> bool:
    """Keep polygon if [min,max] FL overlaps cruise ±pad_fl."""
    lo = props.get("minAlt") or props.get("min_alt")
    hi = props.get("maxAlt") or props.get("max_alt")
    lo_fl = _to_fl(lo)
    hi_fl = _to_fl(hi)
    if lo_fl is None or hi_fl is None:
        return True  # unknown → keep rather than hide useful info
    return (lo_fl - pad_fl) <= cruise_fl <= (hi_fl + pad_fl)

def _intersecting_features(geojson: dict, corridor_poly, cruise_fl: int):
    """Return a filtered GeoJSON with features that intersect the corridor AND overlap cruise FL."""
    if not geojson or not isinstance(geojson, dict) or not corridor_poly:
        return geojson
    feats = []
    for f in geojson.get("features", []):
        try:
            geom = f.get("geometry")
            props = f.get("properties", {}) or {}
            shp = shp_shape(geom)
            if not shp.is_valid:
                continue
            if not shp.intersects(corridor_poly):
                continue
            if not _poly_overlaps_alt(props, cruise_fl):
                continue
            feats.append(f)
        except Exception:
            continue
    return {"type": "FeatureCollection", "features": feats}


def add_geojson_polys(m: folium.Map, gj: dict, name: str, show=True):
    if not gj: return
    fg = folium.FeatureGroup(name=name, show=show, pane="polys_mid")
    def styler(feat):
        props = feat.get("properties", {}) if isinstance(feat, dict) else {}
        hz = props.get("hazard") or props.get("type") or props.get("phenom") or ""
        return _hazard_style(hz)
    folium.GeoJson(gj, name=name, style_function=styler).add_to(fg)
    fg.add_to(m)

# ---- GPT corridor narrative helpers ----

def _meters_route_line(route_coords):
    """Return shapely route LineString in meters (WebMercator) and its length."""
    line_deg = LineString([(lon, lat) for lat, lon in route_coords])
    line_m = shp_transform(WEBMERCATOR, line_deg)
    return line_m, line_m.length

def _minutes_along_route(route_coords, lat, lon, gs_kt=450.0):
    """Approx minutes-into-flight for a point by projecting to the route polyline."""
    line_m, total_m = _meters_route_line(route_coords)
    p_m = shp_transform(WEBMERCATOR, Point(lon, lat))
    s_m = max(0.0, min(line_m.project(p_m), total_m))
    total_nm  = total_m / 1852.0
    total_min = (total_nm / max(gs_kt, 120.0)) * 60.0
    frac = 0.0 if total_m <= 0 else (s_m / total_m)
    return frac * total_min, total_min

def _bucket_minute(t):
    if t < 5:   return "right after takeoff"
    if t < 15:  return "about 10–15 minutes after departure"
    if t < 30:  return "around the 30-minute mark"
    if t < 50:  return "about 45–50 minutes in"
    if t < 75:  return "around the 1-hour point"
    if t < 105: return "about 1 hr 30 min in"
    return f"around {int(round(t/60.0))} hours in"

def analyze_enroute(route_coords, pireps, metars, sig_dom, gairs):
    """Build a compact, model-friendly summary of corridor signals."""
    # PIREPs → timing + levels
    tb_events, ice_events, fls = [], [], []
    for p in pireps or []:
        lat, lon = p.get("lat"), p.get("lon")
        if lat is None or lon is None:
            continue
        tmin, totmin = _minutes_along_route(route_coords, lat, lon)
        tb = (p.get("tbInt1") or p.get("tbInt2") or "").upper()
        ic = (p.get("icgInt1") or p.get("icgInt2") or "").upper()
        if any(k in tb for k in ("LGT","MOD","SEV")):
            lv = "SEV" if "SEV" in tb else ("MOD" if "MOD" in tb else "LGT")
            tb_events.append((round(tmin,1), lv))
        if any(k in ic for k in ("LGT","MOD","SEV")):
            lv = "SEV" if "SEV" in ic else ("MOD" if "MOD" in ic else "LGT")
            ice_events.append((round(tmin,1), lv))
        fl = p.get("fltLvl")
        try:
            if fl:
                fls.append(int(str(fl).replace("FL","").strip()))
        except Exception:
            pass
    tb_events.sort(); ice_events.sort()
    fl_min = min(fls) if fls else None
    fl_max = max(fls) if fls else None

    # En-route METAR quick flags (precip/IFR)
    metar_notes = []
    for row in metars or []:
        wx  = (row.get("wxString") or "").upper()
        cat = (row.get("fltCat") or row.get("flightCategory") or "")
        icao = row.get("icaoId") or ""
        if "TS" in wx or "SHRA" in wx or "RA" in wx or "SN" in wx:
            metar_notes.append({"icao": icao, "wx": wx, "cat": cat})
        elif cat in ("IFR","LIFR"):
            metar_notes.append({"icao": icao, "wx": "", "cat": cat})
        if len(metar_notes) >= 12:
            break

    # Advisory counts
    def _gj_count(gj):
        try: return len((gj or {}).get("features", []))
        except Exception: return 0
    sig_cnt = _gj_count(sig_dom)
    gar_cnt = sum(_gj_count(g) for g in (gairs or []))

    return {
        "tb_events": tb_events,
        "ice_events": ice_events,
        "flight_level_range": [fl_min, fl_max] if fl_min is not None else None,
        "metar_notes": metar_notes,
        "advisories": {"sigmet": sig_cnt, "gairmet": gar_cnt}
    }

def gpt_summary(origin, dest, analysis, total_min, model="gpt-5"):
    prompt = build_captains_user_prompt(origin, dest, total_min, analysis)
    text = call_model_with_retries_for_briefing(prompt, primary_model=model, max_tokens=260)
    return text


def main():
    ap = argparse.ArgumentParser(description="Route corridor + endpoint METAR/TAF demo")
    ap.add_argument("--calm", action="store_true", help="Minimal, passenger-friendly view")
    ap.add_argument("origin", help="Origin ICAO (e.g., KATL)")
    ap.add_argument("destination", help="Destination ICAO (e.g., KDEN)")
    ap.add_argument("--step-nm", type=float, default=50, help="Great-circle sample spacing (NM)")
    ap.add_argument("--buffer-nm", type=float, default=100, help="Route corridor half-width (NM)")
    ap.add_argument("--out", default="route_map.html", help="Output HTML map filename")
    ap.add_argument("--cruise-fl", type=int, default=340, help="Cruise flight level (e.g., 340)")
    ap.add_argument("--pirep-age", type=float, default=1.0, help="PIREPs age window in hours")
    ap.add_argument("--metar-hours", type=float, default=1.0, help="METAR hours window")
    ap.add_argument("--pirep-corridor-buffer-nm", type=float, default=20.0,
                help="How far outside the route corridor to include PIREPs (nautical miles)")

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

    make_map(
        coords, corridor, o, d, o_name, d_name, o_metar, d_metar, o_taf, d_taf,
        out_html=args.out,
        calm=args.calm,
        cruise_fl=args.cruise_fl,
        pirep_age=args.pirep_age,
        metar_hours=args.metar_hours,
        pirep_corridor_buffer_nm=args.pirep_corridor_buffer_nm,
    )


if __name__ == "__main__":
    main()
