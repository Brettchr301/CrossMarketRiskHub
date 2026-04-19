"""Enumerate all 2024-election events from Kalshi via series scan."""
import json
import requests
import time

BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Broader promising series prefixes
promising_series_keywords = ["PRESPARTY", "SENATE", "HOUSE", "GOVPARTY", "GOVERNOR", "POPVOTE", "PRESPOPVOTE", "ECMARGIN", "ELECTORAL", "CONGRESS", "CONTROL", "MARGIN", "PRES"]

# Get all series
r = requests.get(f"{BASE}/series", params={"limit": 10000}, timeout=60)
series = r.json().get("series", [])
print(f"Total series: {len(series)}")

candidates = [s for s in series if any(kw in str(s.get("ticker","")).upper() for kw in promising_series_keywords)]
print(f"Candidate series: {len(candidates)}")

events_2024 = {}  # event_ticker -> {title, series, event_data}
for s in candidates:
    st = s.get("ticker", "")
    try:
        r = requests.get(f"{BASE}/events", params={"series_ticker": st, "limit": 200}, timeout=20)
        if r.status_code != 200:
            continue
        evs = r.json().get("events", [])
        for e in evs:
            et = e.get("event_ticker", "")
            if et.endswith("-24") or et.endswith("-2024") or "24NOV" in et.upper() or "-24NOV" in et:
                events_2024[et] = {
                    "series": st,
                    "title": e.get("title",""),
                    "event": e,
                }
        time.sleep(0.05)
    except Exception as exc:
        print(f"  series {st}: {exc}")

print(f"\n2024 election events found: {len(events_2024)}")
for et, info in sorted(events_2024.items()):
    print(f"  {et:<32} series={info['series']:<20}  {info['title'][:55]}")

# Now for each of these events, call /events/{ticker} with markets included
print("\n\n[Fetching markets per event via /markets?event_ticker=]")
market_counts = {}
total = 0
for et in sorted(events_2024.keys()):
    try:
        r = requests.get(f"{BASE}/markets", params={"event_ticker": et, "limit": 200}, timeout=20)
        if r.status_code == 200:
            mk = r.json().get("markets", [])
            market_counts[et] = len(mk)
            total += len(mk)
        time.sleep(0.05)
    except Exception as exc:
        print(f"  {et}: {exc}")

print(f"\nEvents with markets returned:")
with_markets = {k: v for k, v in market_counts.items() if v > 0}
for et, n in sorted(with_markets.items(), key=lambda kv: -kv[1]):
    print(f"  {et:<32} markets={n}")
print(f"\nTotal events: {len(market_counts)} ({len(with_markets)} with >0 markets)")
print(f"Total markets: {total}")

# Save the discovered list for the backfill step
with open("scripts/kalshi_2024_events.json", "w") as f:
    json.dump({
        "events": list(events_2024.keys()),
        "market_counts": market_counts,
        "titles": {k: v["title"] for k, v in events_2024.items()},
    }, f, indent=2)
print("\nSaved to scripts/kalshi_2024_events.json")
