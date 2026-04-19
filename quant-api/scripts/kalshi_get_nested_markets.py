"""For each 2024 election event, fetch nested markets."""
import json
import requests
import time

BASE = "https://api.elections.kalshi.com/trade-api/v2"

with open("scripts/kalshi_2024_events.json") as f:
    blob = json.load(f)
event_tickers = blob["events"]
print(f"Events to probe: {len(event_tickers)}")

event_market_map = {}
for et in event_tickers:
    # Try with_nested_markets
    try:
        r = requests.get(f"{BASE}/events/{et}", params={"with_nested_markets": "true"}, timeout=20)
        if r.status_code != 200:
            # Try plain
            r = requests.get(f"{BASE}/events/{et}", timeout=20)
        if r.status_code != 200:
            continue
        data = r.json()
        # Two possible shapes
        mk = data.get("markets", []) or data.get("event", {}).get("markets", [])
        if mk:
            event_market_map[et] = mk
    except Exception as exc:
        print(f"  {et}: {exc}")
    time.sleep(0.05)

total = sum(len(v) for v in event_market_map.values())
print(f"Events with markets: {len(event_market_map)} / {len(event_tickers)}")
print(f"Total markets: {total}")
for et, mk in sorted(event_market_map.items(), key=lambda kv: -len(kv[1]))[:30]:
    print(f"  {et:<30} markets={len(mk):3d}  first_ticker={mk[0].get('ticker','')}")

# Save
out = {et: [m.get("ticker") for m in mk] for et, mk in event_market_map.items()}
out_titles = {}
for et, mk in event_market_map.items():
    for m in mk:
        out_titles[m.get("ticker","")] = m.get("title") or m.get("yes_sub_title") or m.get("subtitle") or ""
with open("scripts/kalshi_2024_markets.json", "w") as f:
    json.dump({"events": out, "titles": out_titles}, f, indent=2)
print(f"\nSaved {sum(len(v) for v in out.values())} market tickers to scripts/kalshi_2024_markets.json")
