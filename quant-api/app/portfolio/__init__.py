# Portfolio construction, risk management, and investment decision engine.#
# Modules:
#   robustness.py          - Statistical significance tests (bootstrap, deflated Sharpe, EV, Kelly)
#   risk_manager.py        - Position sizing, circuit breakers, small-portfolio edge
#   quality_filter.py      - ROIC quality screening with McKinsey expectations treadmill
#   translation_edge.py    - Language/accounting/coverage asymmetry scoring
#   portfolio_constructor.py - Greedy rank-based portfolio construction + rebalancing
#   decision_engine.py     - Master orchestration: backtest results -> trade recommendations

from app.portfolio.decision_engine import make_investment_decision, format_decision_report