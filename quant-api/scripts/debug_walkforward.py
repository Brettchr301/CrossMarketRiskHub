"""Debug: trace exactly where tickers are being filtered out."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.backtest.alpha_attribution import SegmentedAlphaBacktester
from app.modeling.global_universe import GLOBAL_COMMODITY_UNIVERSE

bt = SegmentedAlphaBacktester(
    lookback_days=400,
    min_predictions=8,
    signal_threshold=0.001,
    max_tickers=5,
)

universe = GLOBAL_COMMODITY_UNIVERSE[:5]
tickers = [r.ticker for r in universe]
meta = {r.ticker: r for r in universe}
print("Testing tickers:", tickers)

equity_prices, equity_volume = bt._download_prices(tickers, period="6y")
avail = [t for t in tickers if t in equity_prices.columns]
print(f"Available: {avail}")

factor_tickers = [
    "BZ=F", "CL=F", "BOAT", "SEA", "BDRY", "^VIX", "DX-Y.NYB", "^TNX",
    "GC=F", "^IRX", "SI=F", "HG=F", "PL=F", "PA=F", "URA", "LIT",
    "COPX", "REMX", "ALI=F", "WEAT",
]
factor_prices, _ = bt._download_prices(factor_tickers, period="6y")
event_hist = bt._download_event_history()
print(f"Events shape: {event_hist.shape}")

for t in avail[:3]:
    print(f"\n{'='*60}")
    print(f"Testing ticker: {t}")
    row = meta[t]

    frame = bt._build_feature_frame(
        t, equity_prices, equity_volume, factor_prices, event_hist,
        [], {}, 400, "BOAT", "BDRY",
    )
    if frame is None:
        print("  Frame is None!")
        continue

    print(f"  Frame shape: {frame.shape}")
    feature_cols = [c for c in frame.columns if c not in {"y_fwd_20d", "stock_px", "stock_vol"}]
    print(f"  Feature count: {len(feature_cols)}")
    nan_target = frame["y_fwd_20d"].isna().sum()
    print(f"  NaN in target: {nan_target}")
    wf_lookback = bt.walk_forward_lookback
    print(f"  walk_forward_lookback: {wf_lookback}")
    print(f"  len(frame) >= wf_lookback + 40? {len(frame)} >= {wf_lookback + 40} = {len(frame) >= wf_lookback + 40}")

    # Manual walk-forward debug
    all_X = frame[feature_cols].values.astype(np.float64)
    all_Y = frame["y_fwd_20d"].values.astype(np.float64)
    np.nan_to_num(all_X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Feature variance filter
    max_features = 15
    if all_X.shape[1] > max_features:
        variances = np.nanvar(all_X, axis=0)
        top_idx = np.argsort(variances)[-max_features:]
        all_X = all_X[:, top_idx]
    print(f"  After variance filter: X shape = {all_X.shape}")

    lookback = wf_lookback
    step = bt.holding_days
    embargo = 5
    subsample_step = max(1, bt.holding_days // 4)

    print(f"  Loop: range({lookback + embargo}, {len(frame)}, {step})")
    n_iter = 0
    n_nan = 0
    n_toofew = 0
    n_ok = 0
    for i in range(lookback + embargo, len(frame), step):
        n_iter += 1
        train_end = i - embargo
        train_start = max(0, train_end - lookback)
        y_tr = all_Y[train_start:train_end]
        if np.isnan(y_tr).any():
            n_nan += 1
            continue
        y_sub = y_tr[::subsample_step]
        if len(y_sub) < max_features + 1:
            n_toofew += 1
            if n_toofew <= 3:
                print(f"    iter {n_iter}: train_len={train_end-train_start}, subsampled={len(y_sub)}, need={max_features+1}")
            continue
        n_ok += 1

    print(f"  Total iterations: {n_iter}")
    print(f"  NaN-skipped: {n_nan}")
    print(f"  Too-few-samples: {n_toofew}")
    print(f"  Valid windows: {n_ok}")

    # If any valid, try actual _walkforward_harvest
    if n_ok > 0:
        result = bt._walkforward_harvest(frame, "y_fwd_20d")
        if result:
            preds = np.array(result["predictions"])
            print(f"  Predictions: {len(preds)}, range=[{preds.min():.6f}, {preds.max():.6f}]")
            print(f"  Positive (>0.001): {(preds > 0.001).sum()}")
        else:
            print("  _walkforward_harvest returned None despite valid windows!")
    else:
        print("  No valid training windows — cannot produce predictions")
