# Election Prediction Market Historical Data Research

## Context
We have 753,513 historical quotes for 2024 US Presidential/Senate/House markets from Polymarket. We need historical data for 2018, 2020, 2022 elections to backtest cross-platform arbitrage + correlation-alpha strategies.

Current blockers:
1. **Kalshi** has delisted 2024 election markets from public API (returns `markets: []` for all 147 discovered event tickers). We need archival/alternative access.
2. **Polymarket** pre-2024 election markets are sparse in their CLOB API.
3. **PredictIt** has no public history API.
4. **NYT election result archives** (https://static01.nyt.com/elections-assets/pages/data/2024-11-05/results-president.csv) return 404.

## Research Goals

### Goal 1: Kalshi Historical Data Access
Find any of:
- Public Kalshi data archives (Kaggle, Dune Analytics, GitHub dumps)
- Academic datasets (papers on Kalshi arbitrage that published their data)
- pmxt archive coverage for Kalshi (https://archive.pmxt.dev/)
- Third-party Kalshi API mirrors
- Internet Archive snapshots of Kalshi's own data endpoints

Output needed: exact URL(s) + auth requirements + data schema + how to fetch 2024 election candlestick history.

### Goal 2: 2018/2020/2022 Polymarket Data
Find how to access historical price data for:
- 2020 Presidential election (Biden vs Trump)
- 2022 Midterms (Senate control, governor races)

Polymarket CLOB only returns history for markets we can find by ID. Research:
- Is there a Polymarket data dump on HuggingFace / Kaggle / Dune?
- Archive of closed markets beyond gamma-api's 500-per-page pagination limit
- The `archive.pmxt.dev` URL structure for historical Parquet files

### Goal 3: PredictIt Historical Data
Find any public dataset of PredictIt's historical market prices, specifically:
- Academic papers that scraped PredictIt (some published their data)
- Kaggle/GitHub datasets
- ElectionBettingOdds.com data (they aggregate PI + others)
- How to reliably use Wayback Machine for PredictIt's JSON endpoint

### Goal 4: Alternative Prediction Market Data
Cheap/free alternatives that have clean historical data:
- Manifold Markets (has public API with full history)
- Futures on Betfair for US elections (if any)
- BetOnline/Bovada election odds archives

## Output Format
Return a JSON report with:
```json
{
  "kalshi": {
    "archive_urls": ["..."],
    "schema": "...",
    "auth_required": false,
    "coverage_years": [2018, 2020, 2022, 2024],
    "fetch_instructions": "..."
  },
  "polymarket_historical": { ... },
  "predictit_historical": { ... },
  "alternatives": [
    {"name": "Manifold", "url": "...", "coverage": "..."}
  ]
}
```

Use fetch MCP tool to verify URLs return real data. Use Google/Bing web search via fetch to find blog posts, papers, GitHub repos, Kaggle datasets.

Priority: Finding a single reliable archive > comprehensive but unreliable mix.
