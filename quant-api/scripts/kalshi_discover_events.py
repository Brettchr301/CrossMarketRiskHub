"""Step 1: Discover Kalshi 2024 election event tickers."""
import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"

print("=" * 80)
print("Kalshi API Discovery")
print("=" * 80)

# Try events endpoint
print("\n[1] GET /events (settled)")
try:
    r = requests.get(f"{BASE}/events", params={"limit": 200, "status": "settled"}, timeout=30)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        events = data.get("events", [])
        print(f"Total settled events: {len(events)}")
        election_events = [
            e for e in events
            if any(
                kw in (str(e.get("title", "")).lower() + str(e.get("event_ticker", "")).lower())
                for kw in ["election", "senate", "house", "president", "trump", "harris", "biden", "governor"]
            )
        ]
        print(f"Election-like events: {len(election_events)}")
        for e in election_events[:30]:
            print(f"  {e.get('event_ticker', '')}: {str(e.get('title', ''))[:80]}")
    else:
        print(f"Body: {r.text[:300]}")
except Exception as exc:
    print(f"Failed: {exc}")

# Try markets endpoint
print("\n[2] GET /markets (settled)")
try:
    r2 = requests.get(f"{BASE}/markets", params={"limit": 200, "status": "settled"}, timeout=30)
    print(f"Status: {r2.status_code}")
    if r2.status_code == 200:
        markets = r2.json().get("markets", [])
        print(f"Total settled markets returned: {len(markets)}")
        election_markets = [
            m for m in markets
            if any(
                kw in (
                    str(m.get("title", "")).lower()
                    + str(m.get("event_ticker", "")).lower()
                    + str(m.get("ticker", "")).lower()
                )
                for kw in ["pres", "election", "senate", "house", "trump", "harris", "biden", "governor"]
            )
        ]
        print(f"Election-like markets: {len(election_markets)}")
        tickers = set()
        for m in election_markets[:40]:
            et = m.get("event_ticker", "")
            print(f"  {m.get('ticker', '')}  [event={et}]  {str(m.get('title', ''))[:60]}")
            if et:
                tickers.add(et)
        print(f"\nUnique event_tickers (from sample): {sorted(tickers)}")
    else:
        print(f"Body: {r2.text[:300]}")
except Exception as exc:
    print(f"Failed: {exc}")

# Paginate through markets searching for common 2024 election tickers
print("\n[3] Searching known 2024 election tickers directly")
candidate_tickers = [
    "PRES-2024", "PRES24", "PRESPARTY-2024", "PRESWINNER-2024",
    "SENATE-2024", "SENATECTRL-2024", "SENMAJ-24",
    "HOUSE-2024", "HOUSECTRL-2024", "HOUSEMAJ-24",
    "POPVOTE-2024", "GOVERNOR-2024",
    "PRES", "SENATEPARTY", "HOUSEPARTY",
]
found_tickers = {}
for et in candidate_tickers:
    try:
        r = requests.get(
            f"{BASE}/markets", params={"event_ticker": et, "limit": 200}, timeout=20
        )
        if r.status_code == 200:
            mk = r.json().get("markets", [])
            if mk:
                found_tickers[et] = len(mk)
                print(f"  {et}: {len(mk)} markets; first: {mk[0].get('ticker')} - {str(mk[0].get('title',''))[:60]}")
        else:
            pass
    except Exception as exc:
        pass

print(f"\nTickers with markets: {found_tickers}")

# Try /events with election-y tickers/categories
print("\n[4] GET /events with category filter")
for cat in ["Politics", "Elections", "politics", "elections"]:
    try:
        r = requests.get(f"{BASE}/events", params={"limit": 100, "category": cat}, timeout=20)
        if r.status_code == 200:
            evs = r.json().get("events", [])
            print(f"  category={cat}: {len(evs)} events")
            for e in evs[:10]:
                print(f"    {e.get('event_ticker')}: {str(e.get('title',''))[:70]}")
    except Exception as exc:
        print(f"  category={cat}: {exc}")
