import os
from pprint import pprint
from adsb_client import get_aircraft_near

def main():
    center_lat = float(os.getenv("ADSB_CENTER_LAT", "39.8561"))
    center_lon = float(os.getenv("ADSB_CENTER_LON", "-104.6737"))
    radius_nm = float(os.getenv("ADSB_RADIUS_NM", "120"))

    aircraft, meta = get_aircraft_near(
        center_lat,
        center_lon,
        radius_nm,
        airborne_only=True,
        min_alt_ft=10000,
        # Example: airline filters
        # callsign_prefixes=["UAL", "DAL", "SWA", "ASA", "AAL", "FFT"],
    )

    print("=== adsb.lol Pretty Test ===")
    pprint(meta)

    print("\nSample 15 aircraft:")
    for a in aircraft[:15]:
        pprint({
            "callsign": a.callsign,
            "hex": a.hex,
            "type": a.ac_type,
            "reg": a.reg,
            "lat": a.lat,
            "lon": a.lon,
            "alt_baro_ft": a.alt_baro_ft,
            "gs_kt": a.gs_kt,
            "track_deg": a.track_deg,
            "vr_fpm": a.vr_fpm,
            "seen_s": a.seen_s,
            "seen_pos_s": a.seen_pos_s,
            "squawk": a.squawk,
        })
        print("-" * 70)

if __name__ == "__main__":
    main()
