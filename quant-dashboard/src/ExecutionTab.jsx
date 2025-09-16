import { useCallback, useEffect, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8100";

async function fetchJson(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json();
}

function pnlColor(val) {
  return val >= 0 ? "#5fd38d" : "#ff577f";
}

function Badge({ text, variant = "info" }) {
  const colors = {
    info: "#3498db", success: "#2ecc71", warning: "#f1c40f",
    danger: "#e74c3c", muted: "#8da0b5",
  };
  return (
    <span style={{
      background: colors[variant] || colors.info,
      color: "#fff", padding: "2px 8px", borderRadius: 4,
      fontSize: 11, fontWeight: 600, marginLeft: 4,
    }}>
      {text}
    </span>
  );
}

function statusBadge(status) {
  const map = {
    PENDING: "warning", APPROVED: "success", REJECTED: "danger",
    SUBMITTED: "info", FILLED: "success", CANCELLED: "muted",
    EXPIRED: "muted", PAPER_LOGGED: "info", REJECTED_SAFETY: "danger",
    FAILED: "danger", PARTIALLY_APPROVED: "warning", FULLY_APPROVED: "success",
  };
  return <Badge text={status} variant={map[status] || "muted"} />;
}

// ── Portfolio Panel ──────────────────────────────────────────────

function PortfolioPanel({ portfolio }) {
  if (!portfolio || !portfolio.ok) {
    return <section className="panel"><p className="muted">No portfolio data — run IB sync first</p></section>;
  }
  const positions = portfolio.positions || [];
  const sorted = [...positions].sort((a, b) => Math.abs(b.unrealized_pnl) - Math.abs(a.unrealized_pnl));

  return (
    <section className="panel wide">
      <h3>Portfolio Positions ({positions.length})</h3>
      <div className="corr-cards" style={{ marginBottom: 12 }}>
        <div><label>NLV</label><strong>${portfolio.nlv?.toLocaleString()}</strong></div>
        <div><label>Settled Cash</label><strong>${portfolio.settled_cash?.toLocaleString()}</strong></div>
        <div><label>Unsettled</label><strong>${portfolio.unsettled_cash?.toLocaleString()}</strong></div>
        <div><label>Buying Power</label><strong>${portfolio.buying_power?.toLocaleString()}</strong></div>
        <div><label>Source</label><strong>{portfolio.source} {portfolio.is_stale && "(stale)"}</strong></div>
      </div>
      {positions.length === 0 ? (
        <p className="muted">No positions</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Ticker</th><th>Shares</th><th>Avg Cost</th><th>Mkt Price</th>
              <th>Mkt Value</th><th>P&L</th><th>P&L %</th><th>Type</th><th>Days</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((p) => (
              <tr key={p.ticker}>
                <td><strong>{p.ticker}</strong></td>
                <td>{p.shares}</td>
                <td>${p.avg_cost?.toFixed(2)}</td>
                <td>${p.market_price?.toFixed(2)}</td>
                <td>${p.market_value?.toLocaleString()}</td>
                <td style={{ color: pnlColor(p.unrealized_pnl) }}>
                  ${p.unrealized_pnl?.toFixed(0)}
                </td>
                <td style={{ color: pnlColor(p.pnl_pct) }}>
                  {p.pnl_pct?.toFixed(1)}%
                </td>
                <td>{p.commodity_type || "-"}</td>
                <td>{p.days_held ?? "-"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}

// ── Trade Plan Panel ─────────────────────────────────────────────

function TradePlanPanel({ plan, onApprove, onReject, onApproveAll, onRejectAll }) {
  if (!plan || !plan.plan) {
    return <section className="panel"><p className="muted">No active trade plan</p></section>;
  }
  const header = plan.plan;
  const trades = plan.trades || [];
  const rejected = plan.rejected || [];

  const sells = trades.filter((t) => t.action === "CLOSE" || t.action === "REDUCE");
  const buys = trades.filter((t) => t.action === "BUY");

  return (
    <section className="panel wide">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h3>Trade Plan {statusBadge(header.status)}</h3>
        <div>
          <button className="refresh" style={{ marginRight: 6 }}
            onClick={() => onApproveAll(header.plan_id)}>
            Approve All
          </button>
          <button className="refresh" style={{ background: "#e74c3c" }}
            onClick={() => onRejectAll(header.plan_id)}>
            Reject All
          </button>
        </div>
      </div>

      <div className="corr-cards" style={{ marginBottom: 12 }}>
        <div><label>Cash Available</label><strong>${header.cash_available?.toLocaleString()}</strong></div>
        <div><label>Cash After</label><strong>${header.cash_after_trades?.toLocaleString()}</strong></div>
        <div><label>Portfolio Value</label><strong>${header.portfolio_value?.toLocaleString()}</strong></div>
        <div><label>Positions</label><strong>{header.num_positions}</strong></div>
        <div><label>Expires</label><strong>{header.expires_at?.slice(0, 16)}</strong></div>
      </div>

      {sells.length > 0 && (
        <>
          <h4 style={{ color: "#ff577f", marginTop: 8 }}>Sells ({sells.length})</h4>
          <TradeTable trades={sells} onApprove={onApprove} onReject={onReject} />
        </>
      )}

      {buys.length > 0 && (
        <>
          <h4 style={{ color: "#5fd38d", marginTop: 12 }}>Buys — Ranked by EV ({buys.length})</h4>
          <TradeTable trades={buys} onApprove={onApprove} onReject={onReject} />
        </>
      )}

      {rejected.length > 0 && (
        <>
          <h4 style={{ color: "#8da0b5", marginTop: 12 }}>Rejected by Prioritizer ({rejected.length})</h4>
          <table>
            <thead><tr><th>Ticker</th><th>Action</th><th>Reason</th><th>Would Need</th></tr></thead>
            <tbody>
              {rejected.map((r, i) => (
                <tr key={i}>
                  <td>{r.ticker}</td><td>{r.action}</td>
                  <td>{r.reason}</td><td>{r.would_need ? `$${r.would_need.toLocaleString()}` : "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </section>
  );
}

function TradeTable({ trades, onApprove, onReject }) {
  return (
    <table>
      <thead>
        <tr>
          <th>#</th><th>Ticker</th><th>Action</th><th>Shares</th>
          <th>Price</th><th>Cost</th><th>EV%</th><th>Conv</th>
          <th>Flags</th><th>Status</th><th>Actions</th>
        </tr>
      </thead>
      <tbody>
        {trades.map((t) => {
          const flags = typeof t.risk_flags === "string"
            ? JSON.parse(t.risk_flags || "[]") : (t.risk_flags || []);
          return (
            <tr key={t.trade_id}>
              <td>{t.rank}</td>
              <td><strong>{t.ticker}</strong></td>
              <td>{t.action}</td>
              <td>{t.shares}</td>
              <td>${t.limit_price?.toFixed(2)}</td>
              <td>${t.estimated_cost?.toLocaleString()}</td>
              <td style={{ color: pnlColor(t.expected_value_pct) }}>
                {t.expected_value_pct?.toFixed(1)}%
              </td>
              <td>{t.conviction_score?.toFixed(0)}</td>
              <td>{flags.length > 0
                ? flags.map((f, i) => <Badge key={i} text={f} variant="warning" />)
                : "-"}</td>
              <td>{statusBadge(t.status)}</td>
              <td>
                {t.status === "PENDING" && (
                  <>
                    <button className="approve-btn" onClick={() => onApprove(t.trade_id)}
                      style={{ marginRight: 4, background: "#2ecc71", color: "#fff",
                        border: "none", borderRadius: 3, padding: "3px 8px", cursor: "pointer" }}>
                      ✓
                    </button>
                    <button className="reject-btn" onClick={() => onReject(t.trade_id)}
                      style={{ background: "#e74c3c", color: "#fff",
                        border: "none", borderRadius: 3, padding: "3px 8px", cursor: "pointer" }}>
                      ✗
                    </button>
                  </>
                )}
                {t.depends_on_trade_id && <Badge text="T+1" variant="warning" />}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

// ── History Panel ────────────────────────────────────────────────

function HistoryPanel({ history }) {
  const plans = history?.plans || [];
  const trades = history?.trades || [];

  return (
    <section className="panel wide">
      <h3>Trade History</h3>
      {plans.length === 0 ? (
        <p className="muted">No historical plans</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Date</th><th>Plan ID</th><th>Status</th>
              <th>Trades</th><th>Filled</th><th>Portfolio $</th>
            </tr>
          </thead>
          <tbody>
            {plans.slice(0, 20).map((p) => (
              <tr key={p.plan_id}>
                <td>{p.created_at?.slice(0, 10)}</td>
                <td style={{ fontFamily: "monospace", fontSize: 11 }}>{p.plan_id?.slice(0, 12)}</td>
                <td>{statusBadge(p.status)}</td>
                <td>{p.total_trades}</td>
                <td>{p.filled_trades}</td>
                <td>${p.portfolio_value?.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}

// ── Performance Panel ────────────────────────────────────────────

function PerformancePanel({ performance }) {
  const stats = performance?.stats;
  if (!stats || stats.total_trades === 0) {
    return <section className="panel"><p className="muted">No filled trades yet</p></section>;
  }

  return (
    <section className="panel wide">
      <h3>Performance</h3>
      <div className="corr-cards">
        <div><label>Total Trades</label><strong>{stats.total_trades}</strong></div>
        <div><label>Buys / Sells</label><strong>{stats.total_buys} / {stats.total_sells}</strong></div>
        <div>
          <label>Net Realized</label>
          <strong style={{ color: pnlColor(stats.net_realized) }}>
            ${stats.net_realized?.toLocaleString()}
          </strong>
        </div>
        <div><label>Commissions</label><strong>${stats.total_commissions?.toFixed(2)}</strong></div>
      </div>
    </section>
  );
}

// ── Audit Log Panel ──────────────────────────────────────────────

function AuditPanel({ auditLog }) {
  const entries = auditLog?.entries || [];
  return (
    <section className="panel wide">
      <h3>Audit Log (last 50)</h3>
      {entries.length === 0 ? (
        <p className="muted">No audit entries</p>
      ) : (
        <table>
          <thead>
            <tr><th>Time</th><th>Event</th><th>Entity</th><th>Ticker</th><th>Details</th></tr>
          </thead>
          <tbody>
            {entries.slice(0, 50).map((e, i) => (
              <tr key={i}>
                <td style={{ fontSize: 11 }}>{e.timestamp?.slice(0, 19)}</td>
                <td><Badge text={e.event_type} variant={e.user_action ? "success" : "info"} /></td>
                <td>{e.entity_type}:{e.entity_id?.slice(0, 8)}</td>
                <td>{e.ticker || "-"}</td>
                <td style={{ maxWidth: 300, overflow: "hidden", textOverflow: "ellipsis" }}>
                  {e.details?.slice(0, 120)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}

// ── Main Execution Tab ───────────────────────────────────────────

export default function ExecutionTab() {
  const [portfolio, setPortfolio] = useState(null);
  const [tradePlan, setTradePlan] = useState(null);
  const [history, setHistory] = useState(null);
  const [performance, setPerformance] = useState(null);
  const [auditLog, setAuditLog] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [actionMsg, setActionMsg] = useState("");

  const refreshAll = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [port, plan, hist, perf, audit] = await Promise.all([
        fetchJson("/api/execution/portfolio").catch(() => null),
        fetchJson("/api/execution/trade-plan").catch(() => null),
        fetchJson("/api/execution/history?limit=30").catch(() => null),
        fetchJson("/api/execution/performance").catch(() => null),
        fetchJson("/api/execution/audit-log?limit=50").catch(() => null),
      ]);
      setPortfolio(port);
      setTradePlan(plan);
      setHistory(hist);
      setPerformance(perf);
      setAuditLog(audit);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { refreshAll(); }, [refreshAll]);

  const flash = (msg) => { setActionMsg(msg); setTimeout(() => setActionMsg(""), 3000); };

  const handleApprove = async (tradeId) => {
    try {
      const res = await fetchJson("/api/execution/approve", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ trade_ids: [tradeId] }),
      });
      // fetchJson doesn't support POST, inline it
      const resp = await fetch(`${API_BASE}/api/execution/approve`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ trade_ids: [tradeId] }),
      });
      if (!resp.ok) throw new Error(resp.statusText);
      flash(`Approved ${tradeId.slice(0, 8)}`);
      refreshAll();
    } catch (err) { flash(`Error: ${err.message}`); }
  };

  const handleReject = async (tradeId) => {
    try {
      await fetch(`${API_BASE}/api/execution/reject`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ trade_ids: [tradeId], reason: "Rejected from UI" }),
      });
      flash(`Rejected ${tradeId.slice(0, 8)}`);
      refreshAll();
    } catch (err) { flash(`Error: ${err.message}`); }
  };

  const handleApproveAll = async (planId) => {
    try {
      await fetch(`${API_BASE}/api/execution/approve-all?plan_id=${planId}`, { method: "POST" });
      flash("All trades approved");
      refreshAll();
    } catch (err) { flash(`Error: ${err.message}`); }
  };

  const handleRejectAll = async (planId) => {
    try {
      await fetch(`${API_BASE}/api/execution/reject-all?plan_id=${planId}&reason=Bulk+UI+rejection`, { method: "POST" });
      flash("All trades rejected");
      refreshAll();
    } catch (err) { flash(`Error: ${err.message}`); }
  };

  const handleSync = async () => {
    flash("Syncing...");
    try {
      await fetch(`${API_BASE}/api/execution/sync`, { method: "POST" });
      flash("Sync complete");
      refreshAll();
    } catch (err) { flash(`Sync error: ${err.message}`); }
  };

  const handleGeneratePlan = async () => {
    flash("Generating plan...");
    try {
      const resp = await fetch(`${API_BASE}/api/execution/generate-plan`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ force: false }),
      });
      const data = await resp.json();
      flash(data.ok ? `Plan created: ${data.num_trades} trades` : (data.error || "Failed"));
      refreshAll();
    } catch (err) { flash(`Error: ${err.message}`); }
  };

  const handleExecute = async () => {
    if (!window.confirm("Execute approved trades in PAPER mode?")) return;
    flash("Executing...");
    try {
      const resp = await fetch(`${API_BASE}/api/execution/execute`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paper_trade: true }),
      });
      const data = await resp.json();
      flash(data.ok ? `${data.orders_submitted} orders submitted` : (data.error || "Failed"));
      refreshAll();
    } catch (err) { flash(`Error: ${err.message}`); }
  };

  return (
    <div>
      {/* Action Bar */}
      <div style={{
        display: "flex", gap: 8, marginBottom: 16, alignItems: "center",
        padding: "8px 12px", background: "#141b24", borderRadius: 6,
      }}>
        <button className="refresh" onClick={handleSync}>Sync IB</button>
        <button className="refresh" onClick={handleGeneratePlan}>Generate Plan</button>
        <button className="refresh" onClick={handleExecute}
          style={{ background: "#2ecc71" }}>Execute (Paper)</button>
        <button className="refresh" onClick={refreshAll}>Refresh</button>
        {actionMsg && (
          <span style={{ marginLeft: "auto", color: "#ffb020", fontSize: 13 }}>
            {actionMsg}
          </span>
        )}
      </div>

      {loading && <section className="panel">Loading execution data...</section>}
      {error && <section className="panel error">{error}</section>}

      {!loading && !error && (
        <>
          <PortfolioPanel portfolio={portfolio} />
          <TradePlanPanel
            plan={tradePlan}
            onApprove={handleApprove}
            onReject={handleReject}
            onApproveAll={handleApproveAll}
            onRejectAll={handleRejectAll}
          />
          <div className="grid">
            <PerformancePanel performance={performance} />
            <HistoryPanel history={history} />
          </div>
          <AuditPanel auditLog={auditLog} />
        </>
      )}
    </div>
  );
}
