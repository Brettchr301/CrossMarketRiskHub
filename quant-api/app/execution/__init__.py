"""IB Execution & Trade Approval System for CrossMarketRiskHub.

Modules:
  ib_sync           — IB Gateway portfolio sync
  trade_prioritizer — EV-ranked trade prioritization
  approval_queue    — SQLite-backed approval workflow
  order_executor    — IB order submission with safety rails
  notifier          — Discord/email nightly trade plan
  scheduler         — APScheduler daily job orchestration
  decision_bridge   — Cache/load decision signals for execution
"""
