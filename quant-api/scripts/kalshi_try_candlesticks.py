"""Test Kalshi candlesticks endpoint directly with guessed market tickers."""
import requests
from datetime import datetime, timedelta, UTC

BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Common Kalshi outcome suffixes. Binary markets: -YES/-NO; categorical: -D/-R/-O
guesses = [
    # Presidential state examples (Democrat/Republican)
    "PRESPARTYPA-24-D", "PRESPARTYPA-24-R",
    "PRESPARTYPA-24-YES", "PRESPARTYPA-24-NO",
    "PRESPARTYPA-24-DEM", "PRESPARTYPA-24-REP",
    "PRES-2024-TRUMP", "PRES-2024-HARRIS", "PRES-2024-DJT", "PRES-2024-KH",
    "PRES-2024-R", "PRES-2024-D",
    # Senate PA
    "SENATEPA-24-D", "SENATEPA-24-R",
    "SENATEPA-24-CASEY", "SENATEPA-24-MCCORMICK",
    # Popular vote margin
    "POPVOTEMOV-24-T1", "POPVOTEMOV-24-D0.5",
    # Control
    "CONTROLH-2024-D", "CONTROLH-2024-R",
    "CONTROLS-2024-D", "CONTROLS-2024-R",
]

end_ts = int(datetime.now(UTC).timestamp())
start_ts = end_ts - 2 * 365 * 24 * 3600  # 2 years back

# Also try the per-market endpoint /markets/{ticker}
print("\n[Trying /markets/{ticker} and candlesticks]")
for g in guesses:
    try:
        r = requests.get(f"{BASE}/markets/{g}", timeout=10)
        print(f"  /markets/{g}: {r.status_code} {r.text[:120] if r.status_code != 200 else 'OK'}")
        if r.status_code == 200:
            # Try candlesticks
            r2 = requests.get(f"{BASE}/markets/candlesticks", params={
                "market_tickers": g,
                "start_ts": start_ts, "end_ts": end_ts, "period_interval": 1440,
            }, timeout=15)
            print(f"    candlesticks: {r2.status_code}")
            if r2.status_code == 200:
                markets = r2.json().get("markets", [])
                for m in markets:
                    cs = m.get("candlesticks", [])
                    print(f"    candlesticks returned: {len(cs)} points")
    except Exception as exc:
        print(f"  {g}: {exc}")

# Also inspect the series endpoint for PRES series to see its events list
print("\n[GET /series/PRES]")
try:
    r = requests.get(f"{BASE}/series/PRES", timeout=15)
    print(f"  status: {r.status_code}")
    if r.status_code == 200:
        print(f"  body: {r.text[:800]}")
except Exception as exc:
    print(f"  failed: {exc}")

# Try /events/PRES-2024 events endpoint — may include settlement_value & sub_markets
print("\n[GET /events/PRES-2024 full]")
try:
    r = requests.get(f"{BASE}/events/PRES-2024", params={"with_nested_markets": "true"}, timeout=15)
    print(f"  status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"  keys: {list(data.keys())}")
        print(f"  event keys: {list(data.get('event', {}).keys())}")
        print(f"  raw markets field: {data.get('markets')}")
        print(f"  body sample: {r.text[:1000]}")
except Exception as exc:
    print(f"  failed: {exc}")
