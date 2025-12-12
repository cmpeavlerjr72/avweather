import os
import math
import time
import requests
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_BASE = "https://api.adsb.lol"


def nm_to_km(nm: float) -> float:
    return nm * 1.852


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _pick_callsign(obj: Dict[str, Any]) -> Optional[str]:
    for k in ("flight", "call", "callsign", "cs", "ident"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


@dataclass
class Aircraft:
    hex: Optional[str]
    callsign: Optional[str]
    lat: Optional[float]
    lon: Optional[float]
    alt_baro_ft: Optional[float]
    alt_geom_ft: Optional[float]
    gs_kt: Optional[float]
    track_deg: Optional[float]
    vr_fpm: Optional[float]
    reg: Optional[str]
    ac_type: Optional[str]
    squawk: Optional[str]
    seen_s: Optional[float]
    seen_pos_s: Optional[float]
    raw: Dict[str, Any]


def normalize_ac(obj: Dict[str, Any]) -> Aircraft:
    # adsb.lol sample keys match readsb-like feeds:
    # alt_baro, alt_geom, baro_rate, gs, track, flight, hex, lat, lon, r, t, squawk?
    alt_baro = obj.get("alt_baro", obj.get("alt"))
    if isinstance(alt_baro, str) and alt_baro.lower() == "ground":
        alt_baro = None

    return Aircraft(
        hex=(obj.get("hex") or obj.get("icao") or obj.get("icao24")),
        callsign=_pick_callsign(obj),
        lat=obj.get("lat"),
        lon=obj.get("lon"),
        alt_baro_ft=alt_baro,
        alt_geom_ft=obj.get("alt_geom", obj.get("galt")),
        gs_kt=obj.get("gs", obj.get("spd")),
        track_deg=obj.get("track", obj.get("trak")),
        vr_fpm=obj.get("baro_rate", obj.get("vsi")),
        reg=obj.get("r", obj.get("reg")),
        ac_type=obj.get("t", obj.get("type")),
        squawk=obj.get("squawk", obj.get("sqk")),
        seen_s=obj.get("seen"),
        seen_pos_s=obj.get("seen_pos"),
        raw=obj,
    )


def fetch_adsb_lol_near(
    center_lat: float,
    center_lon: float,
    dist_km: float,
    base_url: str = DEFAULT_BASE,
    timeout_s: int = 30,
) -> Dict[str, Any]:
    base_url = base_url.rstrip("/")
    url = f"{base_url}/v2/lat/{center_lat}/lon/{center_lon}/dist/{dist_km}"
    resp = requests.get(url, timeout=timeout_s, headers={"User-Agent": "AvWeather-adsb.lol/1.0"})
    resp.raise_for_status()
    return resp.json()


def get_aircraft_near(
    center_lat: float,
    center_lon: float,
    radius_nm: float,
    *,
    base_url: str = DEFAULT_BASE,
    airborne_only: bool = False,
    min_alt_ft: Optional[float] = None,
    callsign_prefixes: Optional[List[str]] = None,
) -> Tuple[List[Aircraft], Dict[str, Any]]:
    """
    Returns (aircraft_list, meta)
    """
    dist_km = nm_to_km(radius_nm)
    payload = fetch_adsb_lol_near(center_lat, center_lon, dist_km, base_url=base_url)

    raw_list = payload.get("ac") or payload.get("aircraft") or payload.get("planes") or []
    aircraft = [normalize_ac(a) for a in raw_list]

    # Basic sanity filters
    aircraft = [a for a in aircraft if a.lat is not None and a.lon is not None]

    if airborne_only:
        aircraft = [a for a in aircraft if (a.alt_baro_ft is not None) or (a.alt_geom_ft is not None)]

    if min_alt_ft is not None:
        def _alt(a: Aircraft) -> float:
            if a.alt_baro_ft is not None:
                return float(a.alt_baro_ft)
            if a.alt_geom_ft is not None:
                return float(a.alt_geom_ft)
            return -1.0
        aircraft = [a for a in aircraft if _alt(a) >= float(min_alt_ft)]

    if callsign_prefixes:
        prefixes = tuple(p.strip().upper() for p in callsign_prefixes if p.strip())
        aircraft = [a for a in aircraft if (a.callsign or "").upper().startswith(prefixes)]

    meta = {
        "source": "adsb.lol",
        "base_url": base_url,
        "center": {"lat": center_lat, "lon": center_lon},
        "radius_nm": radius_nm,
        "dist_km": dist_km,
        "now": payload.get("now"),
        "total": payload.get("total"),
        "ctime": payload.get("ctime"),
        "ptime": payload.get("ptime"),
        "count_returned": len(aircraft),
    }
    return aircraft, meta
