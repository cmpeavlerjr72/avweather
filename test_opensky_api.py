"""
test_opensky_api.py

Standalone OpenSky API test script.
Used to inspect available live aircraft data before integrating
into the Aviation Forecast Site.

Docs:
https://openskynetwork.github.io/opensky-api/rest.html
"""

import os
import time
import requests
from pprint import pprint
from dotenv import load_dotenv

# --------------------------------------------------
# Setup
# --------------------------------------------------

load_dotenv()

OPENSKY_USERNAME = os.getenv("OPENSKY_USERNAME")
OPENSKY_PASSWORD = os.getenv("OPENSKY_PASSWORD")

if not OPENSKY_USERNAME or not OPENSKY_PASSWORD:
    raise RuntimeError("Missing OPENSKY_USERNAME or OPENSKY_PASSWORD in .env")

BASE_URL = "https://opensky-network.org/api"

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def fetch_all_states():
    """
    Fetch all current aircraft state vectors.
    """
    url = f"{BASE_URL}/states/all"
    resp = requests.get(
        url,
        auth=(OPENSKY_USERNAME, OPENSKY_PASSWORD),
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()


def parse_state(state):
    """
    Convert OpenSky state vector array into a readable dict.
    Index reference:
    https://openskynetwork.github.io/opensky-api/rest.html#response
    """
    return {
        "icao24": state[0],
        "callsign": (state[1] or "").strip(),
        "origin_country": state[2],
        "time_position": state[3],
        "last_contact": state[4],
        "longitude": state[5],
        "latitude": state[6],
        "baro_altitude_m": state[7],
        "on_ground": state[8],
        "velocity_mps": state[9],
        "heading_deg": state[10],
        "vertical_rate_mps": state[11],
        "geo_altitude_m": state[13],
        "squawk": state[14],
        "spi": state[15],
        "position_source": state[16],
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print("Fetching OpenSky live aircraft states...")
    data = fetch_all_states()

    timestamp = data.get("time")
    states = data.get("states", [])

    print(f"\nTimestamp (unix): {timestamp}")
    print(f"Total aircraft returned: {len(states)}")

    if not states:
        print("No aircraft data returned.")
        return

    # Parse states
    parsed = [parse_state(s) for s in states]

    # Example filters (optional)
    airborne = [s for s in parsed if not s["on_ground"]]
    with_callsign = [s for s in airborne if s["callsign"]]

    print(f"\nAirborne aircraft: {len(airborne)}")
    print(f"Airborne w/ callsign: {len(with_callsign)}")

    # Show a sample
    print("\nSample aircraft (first 5):\n")
    for ac in with_callsign[:5]:
        pprint(ac)
        print("-" * 60)

    # Example: airline filter
    print("\nExample: Delta flights (DAL*)\n")
    dal = [s for s in with_callsign if s["callsign"].startswith("DAL")]
    for ac in dal[:5]:
        pprint(ac)
        print("-" * 60)


if __name__ == "__main__":
    main()
