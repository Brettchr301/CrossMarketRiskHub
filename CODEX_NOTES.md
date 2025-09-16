# CrossMarketRiskHub — Implementation Notes for Codex

## Current Status (as of latest session)
- **11/11 tests passing** (`cd quant-api && python3.13 -m pytest tests/ -v`)
- **16/16 API routes working**
- Server runs on port 8100 (`python3.13 -m uvicorn app.main:app --host 127.0.0.1 --port 8100`)
- Global scan endpoint now uses background threading (returns 503 while computing, auto-retries in frontend)

## Already Implemented in This Session
1. `global_scan.py`: Added `scan_async()` background threading so the ~10min first-load doesn't block HTTP
2. `global_scan.py`: Switched walk-forward regression from OLS → **Ridge(alpha=0.8)** (reduces overfitting with 40+ features vs 150 samples)
3. `global_scan.py`: Added cross-features: `brent_ship_cross`, `wti_brent_spread_ret`, `high_vol_regime`, `event_freight_cross`
4. `App.jsx`: Frontend now polls global-opportunities with 10s retry when 503 "computing"
5. Disk cache at `analysis_output/global_opportunity_scan_cache.json` (24hr TTL)

## Next Implementation Priorities (DeepSeek + Claude joint recommendation)

### HIGH PRIORITY — Alpha Generation
1. **New prediction market events to add** in `real_prediction.py` EVENT_MAPPINGS:
   - "OPEC+ cut > 500k bpd at next meeting" (Polymarket slug: `opec-cut-next-meeting`)
   - "Panama Canal daily transits < 30 for 30+ days" (Kalshi)
   - "China stimulus package > 1T yuan" (Kalshi)
   - "US SPR release > 20M barrels" (Kalshi)
   - "US refinery utilization < 85% next month" (Kalshi)

2. **LLM fundamentals extraction** — new file `app/providers/filing_provider.py`:
   - Use Claude API (claude-sonnet-4-6) to parse earnings transcripts
   - Extract structured JSON: production_bpd, opex_per_boe, capex_mm, hedge_ratio, avg_hedge_price, tce_q_guidance
   - Feed into `FundamentalStateBuilder` to replace hardcoded heuristics
   - Cache parsed results in DB (FundamentalStateModel.meta_payload)

3. **Term structure features** — add to `_build_feature_frame()` in global_scan.py:
   ```python
   # Brent 1M-3M spread (contango/backwardation signal)
   if "BZ=F" in factors.columns and "BZG=F" in factors.columns:
       df["brent_term_spread"] = factors["BZ=F"] - factors["BZG=F"]  
   ```

### MEDIUM PRIORITY — Model Improvements
4. **Signal narrative endpoint** — new route `GET /v1/companies/{ticker}/narrative`:
   - Takes signal data + valuation + event probs
   - Calls Claude API to generate 3-sentence investment thesis
   - Returns: thesis, key_risk, top_driver

5. **News pre-emption** — new `app/providers/news_provider.py`:
   - Fetch from GNews or NewsAPI for commodity keywords
   - LLM classifies event relevance + estimates probability shock
   - Triggers `run_event_triggered` pipeline if shock > 3pp

### LOW PRIORITY — Code Hygiene
6. Fix `datetime.utcnow()` → `datetime.now(datetime.UTC)` across all files (deprecation warnings)
7. Add `numpy.errstate(divide='ignore', invalid='ignore')` in `analytics.py` correlation compute

## Running Tests
```bash
cd quant-api
python3.13 -m pytest tests/ -v
```

## Key Architecture Notes
- `ResearchHubService` has its own instance in `GlobalOpportunityService._research_hub` — predictive cache is NOT shared between them. First scan always recomputes predictive contracts (~80s).
- Global scan writes disk cache to `analysis_output/global_opportunity_scan_cache.json` — subsequent server restarts load from disk
- All slow endpoints (predictive-contracts, ticker-research, global-scan) have in-memory caching
