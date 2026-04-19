"""Deep search through Kalshi markets for 2024 election tickers."""
import requests
import time

BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Paginate through /markets broadly filtering for election-y titles
print("=" * 80)
print("Paginating /markets endpoint (all statuses)")
print("=" * 80)

elect_kw = ["pres", "senate", "house", "governor", "election", "trump", "harris", "biden", "popvote", "electoral", "kamala"]

cursor = None
page = 0
all_hits = {}
total_markets = 0
while page < 50:
    params = {"limit": 1000}
    if cursor:
        params["cursor"] = cursor
    try:
        r = requests.get(f"{BASE}/markets", params=params, timeout=30)
        if r.status_code != 200:
            print(f"  page {page}: HTTP {r.status_code}")
            break
        data = r.json()
        mk = data.get("markets", [])
        if not mk:
            break
        total_markets += len(mk)
        for m in mk:
            et = m.get("event_ticker", "")
            title = str(m.get("title", ""))
            ticker = str(m.get("ticker", ""))
            combined = (et + " " + title + " " + ticker).lower()
            # Look for 2024 election indicators
            if any(kw in combined for kw in elect_kw):
                close_time = m.get("close_time", "") or ""
                if "2024" in close_time or "-24" in et or "24NOV" in et.upper():
                    all_hits.setdefault(et, []).append(m)
        cursor = data.get("cursor")
        page += 1
        if not cursor:
            break
        time.sleep(0.1)
    except Exception as exc:
        print(f"  page {page}: {exc}")
        break

print(f"Pages fetched: {page}, total markets scanned: {total_markets}")
print(f"Unique 2024 election event tickers: {len(all_hits)}")
for et, mks in sorted(all_hits.items(), key=lambda kv: -len(kv[1])):
    sample = mks[0]
    print(f"  {et:<30} markets={len(mks):3d}  close={sample.get('close_time','')[:10]}  sample={str(sample.get('title',''))[:55]}")

# Also try series endpoint if available
print("\n\n[Probing /series endpoint]")
try:
    r = requests.get(f"{BASE}/series", params={"limit": 200}, timeout=20)
    print(f"  /series status: {r.status_code}")
    if r.status_code == 200:
        ser = r.json().get("series", [])
        print(f"  Series count: {len(ser)}")
        pol = [s for s in ser if any(kw in str(s).lower() for kw in ["pres", "senate", "house", "elect"])]
        for s in pol[:20]:
            print(f"    {s.get('ticker','?')}  {str(s.get('title',''))[:70]}")
except Exception as exc:
    print(f"  /series: {exc}")

print("\n\n[Probing specific /events/{ticker}]")
# Try known Kalshi 2024 event tickers documented in various public sources
for et in ["PRES-24NOV05", "PRES-2024", "PRESPARTY-24NOV05", "POPVOTE-24NOV05",
           "SENATEPARTY-24NOV05", "HOUSEPARTY-24NOV05",
           "PRESPOPVOTE-24NOV05", "PRESWINNER-24NOV05",
           "ECMARGIN-24NOV05", "TRUMPDJT-24NOV05"]:
    try:
        r = requests.get(f"{BASE}/events/{et}", timeout=15)
        if r.status_code == 200:
            ev = r.json().get("event", {})
            mk = r.json().get("markets", [])
            print(f"  {et}: FOUND -- {ev.get('title','')[:60]}  markets={len(mk)}")
        else:
            print(f"  {et}: {r.status_code}")
    except Exception as exc:
        print(f"  {et}: {exc}")
