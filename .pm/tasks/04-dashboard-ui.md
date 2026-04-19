# Task: Dashboard UI with Minute-Level Price Overlay

## Classification
| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Security | 1 | Serves static files, XSS risk in chart labels |
| Complexity | 3 | Multi-panel interactive charts, real-time data, responsive layout |
| Novelty | 2 | No existing frontend in this project — greenfield React/Vite or vanilla JS |
| Blast Radius | 1 | Additive — new /dashboard route, doesn't change backend logic |
| Existing Code | 1 | Consumes existing API endpoints, adds new data endpoint |
| **Total** | **8** | |

## Objective

Build a single-page dashboard served by FastAPI that visualizes:
1. **Minute-level price overlay** — multiple platforms' prices for a selected race on one chart
2. **Arbitrage heatmap** — time × race matrix showing edge % by color
3. **Alpha model predictions** — bar chart of predicted prob changes with confidence intervals
4. **Race summary table** — sortable table of all tracked races with latest prob, delta, platform count
5. **Weather correlation scatter** — turnout_score vs market price per state

The dashboard must work as a single HTML file served by FastAPI (no separate Node.js server). Use vanilla JavaScript with Chart.js (CDN) for charts — no React build step needed.

## Deliverables

- [ ] `app/election/api/routes.py` — add new data endpoints:
  - `GET /v1/election/chart/price-history/{race_id}` — returns minute-level prices from all platforms for charting
  - `GET /v1/election/chart/arb-heatmap` — returns matrix of {race_id, date, net_edge_pct} for heatmap
  - `GET /v1/election/chart/weather-scatter/{cycle}` — returns {state, turnout_score, market_price, race_id} pairs
- [ ] `app/election/dashboard/index.html` — single-page dashboard with:
  - Header: "Election Alpha Dashboard" + last refresh timestamp
  - Panel 1: Race selector dropdown → minute-level price chart (Chart.js line chart, one series per platform, different colors)
  - Panel 2: Arb heatmap (Chart.js matrix chart or HTML table with color-coded cells)
  - Panel 3: Alpha signals bar chart (horizontal bars, color by confidence)
  - Panel 4: Sortable race summary table (DataTables-like, vanilla JS sort)
  - Panel 5: Weather-price scatter plot (Chart.js scatter, one point per state, labeled)
  - Auto-refresh every 60 seconds via `setInterval`
- [ ] `app/election/api/routes.py` — add `GET /v1/election/dashboard/ui` to serve the HTML file
- [ ] `tests/test_dashboard_endpoints.py` — tests for new data endpoints

## Constraints

- **Single HTML file** — all CSS and JS inline or from CDN (Chart.js 4.x, no npm/webpack)
- **No new Python dependencies** — use FastAPI's `HTMLResponse` and `FileResponse`
- DO NOT create a separate frontend project or package.json
- All chart data comes from API calls (fetch()) — no server-side rendering
- Escape all user-facing strings in JavaScript to prevent XSS (race names, question text)
- Mobile-responsive: CSS grid with `minmax(300px, 1fr)` columns
- Dark theme (background #1a1a2e, text #e0e0e0, chart colors from a predefined palette)

## Exact Interface — New API Endpoints

```python
@router.get("/chart/price-history/{race_id}")
def get_price_history_chart(
    race_id: int,
    hours: int = 168,  # default 1 week
    db: Session = Depends(get_election_db),
) -> dict:
    """Minute-level price history for charting.

    Returns:
    {
        "race_id": 42,
        "race_label": "PA Senate 2024",
        "platforms": {
            "polymarket": [{"ts": "2024-11-04T18:00:00", "price": 0.609}, ...],
            "kalshi": [{"ts": "2024-11-04T18:00:00", "price": 0.621}, ...],
        },
        "outcome": {"winner_party": "D", "election_date": "2024-11-05"},
    }
    """


@router.get("/chart/arb-heatmap")
def get_arb_heatmap(
    cycle: int = 2024,
    db: Session = Depends(get_election_db),
) -> dict:
    """Arbitrage heatmap data.

    Returns:
    {
        "cycle": 2024,
        "races": ["PA Senate", "AZ Governor", ...],
        "dates": ["2024-10-01", "2024-10-02", ...],
        "matrix": [[0.0, 1.2, 0.0, ...], [0.5, 0.0, 2.1, ...], ...],
    }
    """


@router.get("/chart/weather-scatter/{cycle}")
def get_weather_scatter(
    cycle: int,
    db: Session = Depends(get_election_db),
) -> list[dict]:
    """Weather-price correlation scatter data.

    Returns:
    [
        {"state": "PA", "turnout_score": 0.72, "market_price": 0.55, "race_id": 42, "race_label": "PA Senate"},
        ...
    ]
    """
```

## Dashboard HTML Structure

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Election Alpha Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
    <style>
        /* Dark theme, CSS grid layout */
        :root { --bg: #1a1a2e; --card: #16213e; --text: #e0e0e0; --accent: #0f3460; }
        body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
        .card { background: var(--card); border-radius: 12px; padding: 20px; }
        /* ... */
    </style>
</head>
<body>
    <header>
        <h1>Election Alpha Dashboard</h1>
        <span id="last-refresh"></span>
    </header>
    <div class="grid">
        <div class="card" id="price-panel">
            <h2>Price History</h2>
            <select id="race-selector"></select>
            <canvas id="price-chart"></canvas>
        </div>
        <div class="card" id="arb-panel">
            <h2>Arbitrage Heatmap</h2>
            <div id="arb-heatmap"></div>
        </div>
        <div class="card" id="alpha-panel">
            <h2>Alpha Signals</h2>
            <canvas id="alpha-chart"></canvas>
        </div>
        <div class="card" id="races-panel">
            <h2>Race Summary</h2>
            <table id="race-table"><thead>...</thead><tbody></tbody></table>
        </div>
        <div class="card" id="weather-panel">
            <h2>Weather-Price Correlation</h2>
            <canvas id="weather-chart"></canvas>
        </div>
    </div>
    <script>
        // Fetch data from API, build charts, auto-refresh
        const API = '/v1/election';
        // ... Chart.js initialization, data fetching, DOM manipulation
        // ALL string outputs must use textContent (not innerHTML) to prevent XSS
    </script>
</body>
</html>
```

## Tests to Write

1. **test_price_history_endpoint**: Query `/chart/price-history/1` with mocked DB data. Verify response has `platforms` dict with timestamped price arrays.

2. **test_arb_heatmap_endpoint**: Query `/chart/arb-heatmap?cycle=2022`. Verify response has `races`, `dates`, `matrix` with correct dimensions.

3. **test_weather_scatter_endpoint**: Query `/chart/weather-scatter/2024`. Verify response is list of dicts with required keys.

4. **test_dashboard_html_served**: GET `/v1/election/dashboard/ui`. Verify returns 200 with content-type text/html, body contains "Election Alpha Dashboard".

5. **test_empty_data_graceful**: Query all chart endpoints with cycle=9999 (no data). Verify empty but valid responses (no 500 errors).

6. **test_price_history_time_filter**: Query with `hours=24`. Verify all returned timestamps are within last 24 hours of data range.

## Files to Touch
- `app/election/api/routes.py` — modify (add 4 new endpoints)
- `app/election/dashboard/index.html` — create
- `tests/test_dashboard_endpoints.py` — create

## Success Criteria
1. All 6 tests pass
2. `GET /v1/election/dashboard/ui` serves a working HTML page
3. Charts render with real data when API server is running
4. Page auto-refreshes every 60 seconds
5. No XSS vulnerabilities (all dynamic content uses textContent, not innerHTML)
6. Responsive layout works on screens 320px to 1920px wide
7. No new Python or npm dependencies added
