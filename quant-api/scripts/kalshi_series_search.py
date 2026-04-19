"""Use /series listing to find 2024 election-relevant series tickers."""
import requests
import time

BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Get all series and filter
r = requests.get(f"{BASE}/series", params={"limit": 10000}, timeout=60)
data = r.json()
series = data.get("series", [])
print(f"Total series: {len(series)}")

# Filter by title / ticker
kw_list = ["president", "senate", "house", "governor", "electoral", "pres election", "congress", "kamala", "harris", "trump", "biden", "popular vote", "popvote"]
hits = []
for s in series:
    tx = (str(s.get("ticker","")) + " " + str(s.get("title","")) + " " + str(s.get("frequency",""))).lower()
    if any(kw in tx for kw in kw_list):
        hits.append(s)

print(f"Election-like series: {len(hits)}")
for s in hits[:80]:
    print(f"  {s.get('ticker',''):<30} {str(s.get('title',''))[:70]}")

print("\n\n[For candidate series, list events]")
# For the most promising ones, hit /events?series_ticker=
promising = [s.get("ticker") for s in hits if any(kw in str(s.get("ticker","")).upper() for kw in ["PRES", "SENATE", "HOUSE", "GOV", "ELECTORAL", "POPVOTE"])]
promising = list(dict.fromkeys(promising))[:40]

found_2024 = {}
for st in promising:
    try:
        r = requests.get(f"{BASE}/events", params={"series_ticker": st, "limit": 200}, timeout=20)
        if r.status_code != 200:
            continue
        evs = r.json().get("events", [])
        for e in evs:
            et = e.get("event_ticker", "")
            title = str(e.get("title",""))
            close = str(e.get("close_time","") or e.get("strike_date","") or "")
            # Look for 2024 (election date Nov 5, 2024)
            if "2024" in close or "-24" in et or "24NOV" in et.upper() or "24" == et[-2:]:
                found_2024.setdefault(st, []).append(e)
        time.sleep(0.1)
    except Exception as exc:
        print(f"  series {st}: {exc}")

print(f"\nSeries containing 2024 events: {len(found_2024)}")
for st, evs in found_2024.items():
    print(f"\n  Series {st}:")
    for e in evs[:20]:
        print(f"    event={e.get('event_ticker','')}  close={str(e.get('close_time',''))[:10]}  {str(e.get('title',''))[:60]}")
