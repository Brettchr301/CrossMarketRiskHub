import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import ExecutionTab from "./ExecutionTab";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8100";

async function fetchJson(path) {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    throw new Error(`${path} failed: ${response.status}`);
  }
  return response.json();
}

function pct(value) {
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function StatCard({ title, value, subtitle }) {
  return (
    <section className="stat">
      <div className="stat-title">{title}</div>
      <div className="stat-value">{value}</div>
      <div className="stat-sub">{subtitle}</div>
    </section>
  );
}

function DriverRow({ row }) {
  const width = Math.min(100, Math.round(Math.abs(row.correlation) * 100));
  const tone = row.correlation >= 0 ? "pos" : "neg";
  return (
    <div className="driver-row">
      <div className="driver-name">
        {row.name}
        {row.source === "event" ? ` (lag ${row.lag_days}d)` : ""}
      </div>
      <div className="driver-bar-wrap">
        <div className={`driver-bar ${tone}`} style={{ width: `${Math.max(width, 3)}%` }} />
      </div>
      <div className="driver-val">{row.correlation.toFixed(3)}</div>
    </div>
  );
}

/* ── T11: Narrative Side Panel ── */
function NarrativePanel({ ticker, onClose }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!ticker) return;
    let alive = true;
    (async () => {
      try {
        setLoading(true);
        setError("");
        const res = await fetchJson(`/v1/companies/${ticker}/narrative`);
        if (alive) setData(res);
      } catch (err) {
        if (alive) setError(err.message);
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => { alive = false; };
  }, [ticker]);

  if (!ticker) return null;

  return (
    <aside className="narrative-panel">
      <div className="narrative-header">
        <h3>Signal Narrative — {ticker}</h3>
        <button className="narrative-close" onClick={onClose}>×</button>
      </div>
      {loading && <div className="muted">Loading narrative...</div>}
      {error && <div className="error-inline">{error}</div>}
      {data && !loading && (
        <div className="narrative-body">
          <p className="narrative-summary">{data.summary}</p>
          {data.top_drivers?.length > 0 && (
            <div className="narrative-section">
              <h4>Top Drivers</h4>
              <ul className="narrative-drivers">
                {data.top_drivers.map((d, i) => <li key={i}>{d}</li>)}
              </ul>
            </div>
          )}
          {data.risk_note && (
            <div className="narrative-section">
              <h4>Risk</h4>
              <p className="narrative-risk">{data.risk_note}</p>
            </div>
          )}
          {data.contract_context && (
            <div className="narrative-section">
              <h4>Prediction Markets</h4>
              <p>{data.contract_context}</p>
            </div>
          )}
          {data.metrics && (
            <div className="narrative-section">
              <h4>Model Metrics</h4>
              <div className="corr-cards">
                <div><label>Confidence</label><strong>{(data.metrics.confidence * 100).toFixed(1)}%</strong></div>
                <div><label>Hit Rate</label><strong>{(data.metrics.hit_rate * 100).toFixed(1)}%</strong></div>
                <div><label>Score</label><strong>{data.metrics.score}</strong></div>
                <div><label>Oil β</label><strong>{data.metrics.oil_beta?.toFixed(3)}</strong></div>
                <div><label>Ship β</label><strong>{data.metrics.shipping_beta?.toFixed(3)}</strong></div>
                <div><label>Event β</label><strong>{data.metrics.event_beta?.toFixed(3)}</strong></div>
              </div>
            </div>
          )}
        </div>
      )}
    </aside>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState("research");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [downloadError, setDownloadError] = useState("");
  const [downloadingTicker, setDownloadingTicker] = useState("");
  const [events, setEvents] = useState([]);
  const [commodities, setCommodities] = useState([]);
  const [signals, setSignals] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [correlations, setCorrelations] = useState(null);
  const [predictiveContracts, setPredictiveContracts] = useState([]);
  const [selectedTicker, setSelectedTicker] = useState("");
  const [tickerResearch, setTickerResearch] = useState(null);
  const [researchError, setResearchError] = useState("");
  const [researchLoading, setResearchLoading] = useState(false);
  const [globalScan, setGlobalScan] = useState(null);
  const [globalScanLoading, setGlobalScanLoading] = useState(false);
  const [globalScanError, setGlobalScanError] = useState("");
  const [narrativeOpen, setNarrativeOpen] = useState(false);
  const [newsMonitor, setNewsMonitor] = useState(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        setLoading(true);
        setError("");
        const [eventData, commodityData, signalData, metricsData] = await Promise.all([
          fetchJson("/v1/events/probabilities"),
          fetchJson("/v1/commodities/distributions"),
          fetchJson("/v1/signals"),
          fetchJson("/v1/backtest/metrics"),
        ]);
        if (!alive) return;
        setEvents(eventData);
        setCommodities(commodityData);
        setSignals(signalData);
        setMetrics(metricsData);
      } catch (err) {
        if (alive) setError(err.message);
      } finally {
        if (alive) setLoading(false);
      }
    })();

    // Correlations load separately — this endpoint is slow and should not block UI
    (async () => {
      try {
        const correlationData = await fetchJson("/v1/analytics/correlations?lookback_days=260");
        if (alive) setCorrelations(correlationData);
      } catch {
        // Correlations are optional — dashboard works without them
        if (alive) setCorrelations(null);
      }
    })();

    (async () => {
      try {
        const predictive = await fetchJson("/v1/analytics/predictive-contracts?lookback_days=420");
        if (!alive) return;
        setPredictiveContracts(predictive.contracts || []);
      } catch {
        if (alive) setPredictiveContracts([]);
      }
    })();

    (async () => {
      setGlobalScanLoading(true);
      setGlobalScanError("");
      const url = "/v1/analytics/global-opportunities?lookback_days=780&min_modeled_count=200&max_rows=220";
      // Poll until scan is ready (202 = still computing in background)
      for (let attempt = 0; attempt < 30 && alive; attempt++) {
        try {
          const globalData = await fetchJson(url);
          if (!alive) return;
          setGlobalScan(globalData);
          setGlobalScanLoading(false);
          return;
        } catch (err) {
          if (!alive) return;
          if (err.message && (err.message.includes("503") || err.message.includes("computing"))) {
            // Still computing — wait and retry
            await new Promise((r) => setTimeout(r, 10000));
          } else {
            setGlobalScanError(err.message);
            setGlobalScanLoading(false);
            return;
          }
        }
      }
      if (alive) {
        setGlobalScanError("Global scan timed out — server is still computing");
        setGlobalScanLoading(false);
      }
    })();

    return () => {
      alive = false;
    };
  }, []);

  /* ── T12: News Monitor Status badge — poll every 60s ── */
  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const status = await fetchJson("/v1/news/monitor-status");
        if (alive) setNewsMonitor(status);
      } catch {
        if (alive) setNewsMonitor(null);
      }
    };
    poll();
    const interval = setInterval(poll, 60_000);
    return () => { alive = false; clearInterval(interval); };
  }, []);

  const topSignals = useMemo(() => {
    const seen = new Set();
    const deduped = [];
    for (const row of [...signals].sort((a, b) => b.score - a.score)) {
      if (seen.has(row.ticker)) continue;
      seen.add(row.ticker);
      deduped.push(row);
      if (deduped.length >= 12) break;
    }
    return deduped;
  }, [signals]);

  const avgProb = useMemo(() => {
    if (events.length === 0) return 0;
    return events.reduce((sum, row) => sum + Number(row.prob || 0), 0) / events.length;
  }, [events]);

  const eventChartData = useMemo(
    () => events.map((e) => ({ event: e.event_id, prob: Number((e.prob * 100).toFixed(2)) })),
    [events],
  );

  const commodityChartData = useMemo(
    () =>
      [...commodities]
        .sort((a, b) => a.symbol.localeCompare(b.symbol))
        .map((c) => ({
          symbol: c.symbol,
          p05: Number(c.p05),
          p50: Number(c.p50),
          p95: Number(c.p95),
        })),
    [commodities],
  );

  const signalScatterData = useMemo(
    () =>
      topSignals.map((s) => ({
        ticker: s.ticker,
        score: Number(s.score),
        net: Number(s.expected_return_net_cost) * 100,
      })),
    [topSignals],
  );

  const correlationMap = useMemo(() => {
    const rows = correlations?.tickers ?? [];
    const out = {};
    for (const row of rows) out[row.ticker] = row;
    return out;
  }, [correlations]);

  const globalOpportunities = useMemo(() => globalScan?.opportunities ?? [], [globalScan]);
  const commodityTypeStats = useMemo(() => globalScan?.commodity_type_stats ?? [], [globalScan]);
  const tickerOptions = useMemo(() => {
    const out = [];
    const seen = new Set();
    for (const row of topSignals) {
      if (!seen.has(row.ticker)) {
        seen.add(row.ticker);
        out.push(row.ticker);
      }
    }
    for (const row of globalOpportunities) {
      if (!seen.has(row.ticker)) {
        seen.add(row.ticker);
        out.push(row.ticker);
      }
    }
    return out;
  }, [topSignals, globalOpportunities]);

  useEffect(() => {
    if (selectedTicker) return;
    const fallback = tickerOptions[0];
    if (fallback) setSelectedTicker(fallback);
  }, [selectedTicker, tickerOptions]);

  useEffect(() => {
    if (!selectedTicker) return;
    let alive = true;
    (async () => {
      try {
        setResearchLoading(true);
        setResearchError("");
        const data = await fetchJson(`/v1/analytics/tickers/${selectedTicker}/research?lookback_days=260`);
        if (!alive) return;
        setTickerResearch(data);
      } catch (err) {
        if (alive) setResearchError(err.message);
      } finally {
        if (alive) setResearchLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, [selectedTicker]);

  const selectedCorr = selectedTicker ? correlationMap[selectedTicker] : null;

  const researchChartData = useMemo(() => tickerResearch?.series ?? [], [tickerResearch]);
  const commodityTypeChartData = useMemo(
    () =>
      commodityTypeStats.slice(0, 12).map((row) => ({
        commodity_type: row.commodity_type,
        net: Number(row.top_bucket_avg_net_return) * 100,
        hit: Number(row.avg_hit_rate) * 100,
      })),
    [commodityTypeStats],
  );
  const selectedTopContracts = useMemo(
    () =>
      tickerResearch?.top_predictive_contracts?.length
        ? tickerResearch.top_predictive_contracts
        : predictiveContracts.filter((c) => c.best_target === selectedTicker).slice(0, 10),
    [tickerResearch, predictiveContracts, selectedTicker],
  );

  async function downloadWorkbook(ticker) {
    try {
      setDownloadError("");
      setDownloadingTicker(ticker);
      const response = await fetch(`${API_BASE}/v1/companies/${ticker}/valuation-distribution.xlsx`);
      if (!response.ok) {
        throw new Error(`Download failed (${response.status})`);
      }
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `${ticker}_valuation_distribution.xlsx`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setDownloadError(`${ticker}: ${err.message}`);
    } finally {
      setDownloadingTicker("");
    }
  }

  return (
    <main className="app-shell">
      <header className="hero">
        <div>
          <h1>Market Risk Workstation</h1>
          <p>Trading + research hub for prediction markets, options, futures, freight, and valuation diagnostics.</p>
        </div>
        <div className="hero-controls">
          {/* Tab Switcher */}
          <nav style={{ display: "flex", gap: 2, marginRight: 12 }}>
            {[["research", "Research"], ["execution", "Execution"]].map(([key, label]) => (
              <button
                key={key}
                onClick={() => setActiveTab(key)}
                style={{
                  padding: "6px 16px", border: "none", borderRadius: 4, cursor: "pointer",
                  fontWeight: 600, fontSize: 13,
                  background: activeTab === key ? "#ffb020" : "#1e2a38",
                  color: activeTab === key ? "#0d1117" : "#8da0b5",
                }}
              >
                {label}
              </button>
            ))}
          </nav>
          {/* T12: News Monitor Badge */}
          <span
            className="pill"
            title={newsMonitor ? `Last poll: ${newsMonitor.last_poll ?? "never"} | Events triggered: ${newsMonitor.events_triggered}` : "News monitor unavailable"}
            style={{ display: "inline-flex", alignItems: "center", gap: 6 }}
          >
            <span
              style={{
                width: 8, height: 8, borderRadius: "50%",
                background: newsMonitor?.running ? "#5fd38d" : "#ff577f",
                display: "inline-block",
              }}
            />
            News {newsMonitor?.running ? "Live" : "Off"}
          </span>
          <span className="pill">Desktop Ready</span>
          <button className="refresh" onClick={() => window.location.reload()}>
            Refresh
          </button>
        </div>
      </header>

      {/* Execution Tab */}
      {activeTab === "execution" && <ExecutionTab />}

      {/* Research Tab (original content) */}
      {activeTab === "research" && loading && <section className="panel">Loading data feeds and analytics...</section>}
      {activeTab === "research" && error && <section className="panel error">{error}</section>}

      {activeTab === "research" && !loading && !error && (
        <>
          <section className="stats">
            <StatCard title="Tracked Event Themes" value={events.length} subtitle={`Avg Prob ${pct(avgProb)}`} />
            <StatCard title="Signal Candidates" value={topSignals.length} subtitle="Net-of-cost filtered" />
            <StatCard
              title="Backtest IRR"
              value={metrics ? pct(metrics.irr) : "N/A"}
              subtitle={metrics ? `Sharpe ${Number(metrics.sharpe).toFixed(2)}` : "No metrics"}
            />
            <StatCard
              title="Predictive Contracts"
              value={predictiveContracts.length}
              subtitle={`Correlation rows ${correlations?.tickers?.length ?? 0}`}
            />
            <StatCard
              title="Global Modeled"
              value={globalScan?.modeled_count ?? "N/A"}
              subtitle={`Universe ${globalScan?.universe_size ?? 0}`}
            />
          </section>

          <section className="grid">
            <section className="panel chart-panel">
              <h3>Event Probability Map</h3>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={eventChartData} layout="vertical" margin={{ left: 10, right: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#25303d" />
                  <XAxis type="number" domain={[0, 100]} stroke="#8da0b5" />
                  <YAxis dataKey="event" type="category" width={150} stroke="#8da0b5" />
                  <Tooltip formatter={(value) => `${value}%`} />
                  <Bar dataKey="prob" radius={[0, 4, 4, 0]}>
                    {eventChartData.map((entry) => (
                      <Cell key={entry.event} fill={entry.prob >= 50 ? "#ffb020" : "#37c2d0"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </section>

            <section className="panel chart-panel">
              <h3>Commodity / Freight Distribution Bands</h3>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={commodityChartData} margin={{ top: 5, right: 10, left: 5, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#25303d" />
                  <XAxis dataKey="symbol" stroke="#8da0b5" />
                  <YAxis stroke="#8da0b5" />
                  <Tooltip />
                  <Line type="monotone" dataKey="p05" stroke="#4ca6ff" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="p50" stroke="#ffb020" strokeWidth={2.5} dot={false} />
                  <Line type="monotone" dataKey="p95" stroke="#ff577f" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </section>

            <section className="panel chart-panel">
              <h3>Signal Score vs Net Return</h3>
              <ResponsiveContainer width="100%" height={260}>
                <ScatterChart margin={{ top: 10, right: 15, bottom: 10, left: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#25303d" />
                  <XAxis type="number" dataKey="score" name="Score" stroke="#8da0b5" />
                  <YAxis type="number" dataKey="net" name="Net Return %" stroke="#8da0b5" />
                  <Tooltip cursor={{ strokeDasharray: "3 3" }} formatter={(v) => Number(v).toFixed(2)} />
                  <Scatter data={signalScatterData} fill="#37c2d0" />
                </ScatterChart>
              </ResponsiveContainer>
            </section>

            <section className="panel correlation-panel">
              <div className="correlation-head">
                <h3>Ticker Correlation Explorer</h3>
                <select value={selectedTicker} onChange={(e) => setSelectedTicker(e.target.value)}>
                  {tickerOptions.map((ticker) => (
                    <option key={ticker} value={ticker}>
                      {ticker}
                    </option>
                  ))}
                </select>
              </div>
              {selectedCorr ? (
                <>
                  <div className="corr-cards">
                    <div>
                      <label>Brent Corr</label>
                      <strong>{selectedCorr.corr_brent?.toFixed(3) ?? "N/A"}</strong>
                    </div>
                    <div>
                      <label>WTI Corr</label>
                      <strong>{selectedCorr.corr_wti?.toFixed(3) ?? "N/A"}</strong>
                    </div>
                    <div>
                      <label>Shipping Corr</label>
                      <strong>{selectedCorr.corr_shipping?.toFixed(3) ?? "N/A"}</strong>
                    </div>
                    <div>
                      <label>Samples</label>
                      <strong>{selectedCorr.sample_size}</strong>
                    </div>
                  </div>
                  <div className="driver-list">
                    {selectedCorr.top_drivers.map((d) => (
                      <DriverRow key={`${d.source}-${d.name}-${d.lag_days}`} row={d} />
                    ))}
                  </div>
                </>
              ) : (
                <div className="muted">No correlation data for selected ticker.</div>
              )}
            </section>

            <section className="panel wide chart-panel">
              <div className="correlation-head">
                <h3>Per-Stock Model Inputs vs Price ({selectedTicker || "Select"})</h3>
                {researchLoading && <span className="pill">Computing...</span>}
              </div>
              {researchError && <div className="error-inline">{researchError}</div>}
              {!researchError && researchChartData.length > 0 && (
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={researchChartData} margin={{ top: 5, right: 10, left: 5, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#25303d" />
                    <XAxis dataKey="date" hide />
                    <YAxis stroke="#8da0b5" />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="stock" stroke="#ffb020" dot={false} strokeWidth={2.2} />
                    <Line type="monotone" dataKey="brent" stroke="#37c2d0" dot={false} strokeWidth={1.8} />
                    <Line type="monotone" dataKey="wti" stroke="#4ca6ff" dot={false} strokeWidth={1.6} />
                    <Line type="monotone" dataKey="shipping_spot" stroke="#9a89ff" dot={false} strokeWidth={1.4} />
                    <Line type="monotone" dataKey="shipping_fwd" stroke="#5fd38d" dot={false} strokeWidth={1.4} />
                    <Line type="monotone" dataKey="event_oil_100" stroke="#ff668e" dot={false} strokeWidth={1.2} />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </section>

            <section className="panel">
              <h3>Model Validation</h3>
              {tickerResearch?.validation ? (
                <div className="corr-cards">
                  <div>
                    <label>Baseline Hit</label>
                    <strong>{pct(tickerResearch.validation.baseline_hit_rate)}</strong>
                  </div>
                  <div>
                    <label>Enriched Hit</label>
                    <strong>{pct(tickerResearch.validation.enriched_hit_rate)}</strong>
                  </div>
                  <div>
                    <label>Fair Value</label>
                    <strong>${Number(tickerResearch.validation.fair_value_price).toFixed(2)}</strong>
                  </div>
                  <div>
                    <label>Spot</label>
                    <strong>${Number(tickerResearch.validation.spot_price).toFixed(2)}</strong>
                  </div>
                </div>
              ) : (
                <div className="muted">No validation snapshot yet.</div>
              )}
            </section>

            <section className="panel">
              <h3>Shipping Hedge Lens</h3>
              {tickerResearch?.hedge ? (
                <div className="corr-cards">
                  <div>
                    <label>Spot Proxy</label>
                    <strong>{tickerResearch.hedge.spot_proxy}</strong>
                  </div>
                  <div>
                    <label>Forward Proxy</label>
                    <strong>{tickerResearch.hedge.forward_proxy}</strong>
                  </div>
                  <div>
                    <label>Current Basis</label>
                    <strong>{pct(tickerResearch.hedge.current_basis_pct)}</strong>
                  </div>
                  <div>
                    <label>1M Expected Basis</label>
                    <strong>{pct(tickerResearch.hedge.one_month_expected_basis_pct)}</strong>
                  </div>
                </div>
              ) : (
                <div className="muted">No hedge stats.</div>
              )}
            </section>

            <section className="panel wide">
              <h3>Top Predictive Contracts For {selectedTicker || "Ticker"}</h3>
              <table>
                <thead>
                  <tr>
                    <th>Contract</th>
                    <th>Category</th>
                    <th>Target</th>
                    <th>Lead</th>
                    <th>Corr</th>
                    <th>Liquidity</th>
                    <th>Score</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedTopContracts.length === 0 ? (
                    <tr>
                      <td colSpan={7} className="muted">
                        No contract diagnostics available.
                      </td>
                    </tr>
                  ) : (
                    selectedTopContracts.map((row) => (
                      <tr key={row.market_id}>
                        <td>{row.question}</td>
                        <td>{row.category}</td>
                        <td>{row.best_target}</td>
                        <td>{row.lead_days}d</td>
                        <td>{Number(row.correlation).toFixed(3)}</td>
                        <td>{Number(row.liquidity_score).toFixed(3)}</td>
                        <td>{Number(row.predictive_score).toFixed(3)}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </section>

            <section className="panel chart-panel">
              <h3>Commodity-Type Effectiveness</h3>
              {commodityTypeChartData.length > 0 ? (
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={commodityTypeChartData} margin={{ top: 5, right: 10, left: 5, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#25303d" />
                    <XAxis dataKey="commodity_type" stroke="#8da0b5" angle={-20} textAnchor="end" interval={0} height={80} />
                    <YAxis stroke="#8da0b5" />
                    <Tooltip formatter={(v) => `${Number(v).toFixed(2)}%`} />
                    <Legend />
                    <Bar dataKey="net" name="Top Bucket Net %" fill="#37c2d0" />
                    <Bar dataKey="hit" name="Avg Hit %" fill="#ffb020" />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="muted">{globalScanLoading ? "Running global scan..." : "No type stats yet."}</div>
              )}
            </section>

            <section className="panel wide">
              <h3>Global 200+ Mispricing Scanner</h3>
              {globalScanError && <div className="error-inline">{globalScanError}</div>}
              {!globalScanError && globalScanLoading && <div className="muted">Running global cross-market scan...</div>}
              {!globalScanError && globalOpportunities.length > 0 && (
                <table>
                  <thead>
                    <tr>
                      <th>Ticker</th>
                      <th>Type</th>
                      <th>Country</th>
                      <th>Dir</th>
                      <th>Score</th>
                      <th>Net</th>
                      <th>Fair vs Spot</th>
                      <th>Hit</th>
                      <th>Margin d</th>
                      <th>Contracts</th>
                    </tr>
                  </thead>
                  <tbody>
                    {globalOpportunities.slice(0, 140).map((row) => (
                      <tr
                        key={`global-${row.ticker}`}
                        onClick={() => { setSelectedTicker(row.ticker); setNarrativeOpen(true); }}
                        style={{ cursor: "pointer" }}
                      >
                        <td>{row.ticker}</td>
                        <td>{row.commodity_type}</td>
                        <td>{row.country}</td>
                        <td>{row.direction}</td>
                        {/* T13: Ridge Alpha Tooltip on Score */}
                        <td title={row.ridge_alpha != null ? `Ridge α = ${row.ridge_alpha}` : undefined}>
                          {Number(row.score).toFixed(2)}
                        </td>
                        <td>{pct(row.expected_return_net_cost)}</td>
                        <td>
                          ${Number(row.fair_value_price).toFixed(2)} / ${Number(row.spot_price).toFixed(2)}
                        </td>
                        <td>{pct(row.hit_rate)}</td>
                        <td>{pct(row.predicted_margin_change)}</td>
                        <td>{(row.top_predictive_contracts || []).slice(0, 2).join(" | ") || "-"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </section>

            <section className="panel wide">
              <h3>Signal Board + Valuation Exports</h3>
              {downloadError && <div className="error-inline">{downloadError}</div>}
              <table>
                <thead>
                  <tr>
                    <th>Ticker</th>
                    <th>Direction</th>
                    <th>Score</th>
                    <th>Net Return</th>
                    <th>Risk Flags</th>
                    <th>Workbook</th>
                  </tr>
                </thead>
                <tbody>
                  {topSignals.map((row) => (
                    <tr key={row.ticker}>
                      <td>{row.ticker}</td>
                      <td>{row.direction}</td>
                      <td>{Number(row.score).toFixed(2)}</td>
                      <td>{pct(row.expected_return_net_cost)}</td>
                      <td>{(row.risk_flags || []).join(", ") || "-"}</td>
                      <td>
                        <button
                          className="download-btn"
                          onClick={() => downloadWorkbook(row.ticker)}
                          disabled={downloadingTicker === row.ticker}
                        >
                          {downloadingTicker === row.ticker ? "Downloading..." : "Download XLSX"}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          </section>

          {/* T11: Narrative Side Panel */}
          {narrativeOpen && selectedTicker && (
            <NarrativePanel ticker={selectedTicker} onClose={() => setNarrativeOpen(false)} />
          )}
        </>
      )}
    </main>
  );
}
