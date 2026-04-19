"""Paginate through Kalshi events/markets to find 2024 election tickers."""
import requests
import time

BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Try direct ticker candidates based on Kalshi naming conventions observed
# Pattern: KXTICKER-YY or TICKER-YY, e.g. PRES-24, SENATE-24
candidates = [
    # Presidential
    "PRES-24", "PRES-24NOV05", "PRESPARTY-24", "PRESPOPVOTE-24",
    "KXPRES-24", "KXPRESPARTY-24", "KXPRESPOPVOTE-24",
    # Senate / House
    "SENATE-24", "SENATECTRL-24", "SENATEMAJ-24", "KXSENATE-24",
    "HOUSE-24", "HOUSECTRL-24", "HOUSEMAJ-24", "KXHOUSE-24",
    # State-level senates (sample)
    "SENATEOH-24", "SENATEPA-24", "SENATEAZ-24", "SENATEWI-24", "SENATEMI-24",
    "SENATENV-24", "SENATEMT-24", "SENATETX-24", "SENATEFL-24",
    # Governor
    "GOVERNOR-24", "GOVERNORNC-24", "GOVERNORWA-24",
    "GOVPARTYNC-24", "GOVPARTYWA-24",
    # Popular vote / swing states
    "PRESWINNER-24", "PRESPOPVOTE-24", "KXPRESPOPVOTE-24",
    "PRESPA-24", "PRESMI-24", "PRESWI-24", "PRESAZ-24", "PRESGA-24", "PRESNV-24", "PRESNC-24",
    "KXPRESPA-24", "KXPRESMI-24", "KXPRESWI-24",
    # Candidate-specific
    "TRUMPWIN-24", "HARRISWIN-24", "BIDENWIN-24",
    # Other plausible formats
    "PRES2024", "SENATE2024", "HOUSE2024",
    "PRES", "SENATEPARTY", "HOUSEPARTY",
    # Electoral college
    "ECMARGIN-24", "ELECTORALCOLLEGE-24",
]

print("=" * 80)
print("Probing candidate 2024 event tickers")
print("=" * 80)

found = {}
for et in candidates:
    try:
        r = requests.get(
            f"{BASE}/markets", params={"event_ticker": et, "limit": 200, "status": "settled"}, timeout=20
        )
        if r.status_code == 200:
            markets = r.json().get("markets", [])
            if markets:
                found[et] = markets
                print(f"  FOUND {et}: {len(markets)} markets (first: {markets[0].get('ticker')}) - {str(markets[0].get('title',''))[:60]}")
        r2 = requests.get(
            f"{BASE}/markets", params={"event_ticker": et, "limit": 200}, timeout=20
        )
        if r2.status_code == 200:
            markets2 = r2.json().get("markets", [])
            if markets2 and et not in found:
                found[et] = markets2
                print(f"  FOUND (any status) {et}: {len(markets2)} markets")
        time.sleep(0.15)
    except Exception as exc:
        print(f"  {et}: {exc}")

print(f"\n\nTotal tickers with markets: {len(found)}")
for et, mks in found.items():
    print(f"  {et}: {len(mks)}")

# Now paginate through /events looking at anything suffix "-24"
print("\n\n[Paginating /events for suffix containing '24']")
cursor = None
page = 0
all_2024 = []
while page < 20:
    params = {"limit": 200, "with_nested_markets": "false"}
    if cursor:
        params["cursor"] = cursor
    try:
        r = requests.get(f"{BASE}/events", params=params, timeout=30)
        if r.status_code != 200:
            print(f"  page {page}: status {r.status_code}")
            break
        data = r.json()
        evs = data.get("events", [])
        if not evs:
            break
        for e in evs:
            et = e.get("event_ticker", "")
            if ("-24" in et and "-240" not in et and "-245" not in et) or et.endswith("24"):
                all_2024.append(e)
        cursor = data.get("cursor")
        page += 1
        if not cursor:
            break
    except Exception as exc:
        print(f"  page {page}: {exc}")
        break

print(f"Pages fetched: {page}")
print(f"Events with '-24' suffix: {len(all_2024)}")
elect_kw = ["pres", "senate", "house", "governor", "election", "trump", "harris", "biden", "popvote", "electoral"]
filtered = [e for e in all_2024 if any(kw in (str(e.get("title","")).lower() + str(e.get("event_ticker","")).lower()) for kw in elect_kw)]
print(f"After election-keyword filter: {len(filtered)}")
for e in filtered[:60]:
    print(f"  {e.get('event_ticker'):<30} {str(e.get('title',''))[:70]}")
