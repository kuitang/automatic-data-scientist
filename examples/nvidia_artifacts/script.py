#!/usr/bin/env python3
import argparse
import sys
import os
import io
import warnings
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

def find_col(df, candidates):
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in cols_lower:
            return cols_lower[key]
    return None

def format_currency(x):
    try:
        return f"${x:,.2f}"
    except Exception:
        return str(x)

def format_percent(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "NA"
        return f"{x*100:.2f}%"
    except Exception:
        return str(x)

def svg_from_fig(fig, dpi=80):
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    svg_data = buf.read().decode('utf-8')
    return svg_data

def compute_drawdown(equity):
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return dd

def annualized_sharpe(daily_returns, periods_per_year=252):
    r = pd.Series(daily_returns).dropna()
    if r.std(ddof=1) == 0 or len(r) < 2:
        return np.nan
    return np.sqrt(periods_per_year) * r.mean() / r.std(ddof=1)

def get_ma_windows(n):
    # Adaptive windows by sample length; fixed ex-ante within each period (no look-ahead)
    if n >= 400:
        sw, lw = 20, 100
    elif n >= 200:
        sw, lw = 15, 75
    elif n >= 120:
        sw, lw = 10, 50
    elif n >= 60:
        sw, lw = 8, 30
    else:
        sw = max(5, n // 8) if n >= 10 else 2
        lw = max(sw + 3, n // 3 if n >= 9 else sw + 3)
    return sw, lw

def get_mom_lookback(n):
    # Adaptive lookback by sample length; fixed ex-ante within each period (no look-ahead)
    if n >= 200:
        return 60
    elif n >= 120:
        return 40
    elif n >= 60:
        return 20
    else:
        return max(5, n // 4) if n >= 12 else 3

def signal_buy_and_hold(prices, params=None):
    return pd.Series(1.0, index=prices.index)

def signal_ma_trend(prices, params):
    sw = int(params.get("short_window", 20))
    lw = int(params.get("long_window", 100))
    s = prices.rolling(window=sw, min_periods=sw).mean()
    l = prices.rolling(window=lw, min_periods=lw).mean()
    sig = (s > l).astype(float)
    sig = sig.fillna(0.0)
    return sig

def signal_tsmom(prices, params):
    look = int(params.get("lookback", 60))
    mom = prices / prices.shift(look) - 1.0
    sig = (mom > 0).astype(float)
    sig = sig.fillna(0.0)
    return sig

def backtest(prices, transaction_cost=0.0005, strategy="bh", params=None, initial=1000.0):
    if params is None:
        params = {}
    # Daily simple returns from Adj Close
    ret = prices.pct_change().fillna(0.0)

    if strategy == "bh":
        sig = signal_buy_and_hold(prices)
    elif strategy == "ma":
        sig = signal_ma_trend(prices, params)
    elif strategy == "tsmom":
        sig = signal_tsmom(prices, params)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Use previous information: apply today's position based on yesterday's signal
    pos = sig.shift(1).fillna(0.0).clip(0.0, 1.0)
    trades = pos.diff().abs().fillna(pos.abs())  # fraction of portfolio traded at start of day

    # Apply transaction cost when position changes at start of day, then apply returns
    # equity_t = equity_{t-1} * (1 - cost*trade_t) * (1 + pos_t * ret_t)
    factors = ((1.0 - transaction_cost * trades).clip(lower=0.0) * (1.0 + pos * ret)).astype(float)
    equity = initial * factors.cumprod()
    strat_returns = factors - 1.0  # daily strategy returns

    return {
        "equity": equity,
        "returns": strat_returns,
        "position": pos,
        "signal": sig,
        "trades": trades
    }

def split_periods(df):
    n = len(df)
    i1 = n // 3
    i2 = 2 * n // 3
    return {
        "Full sample": df,
        "Early third": df.iloc[:i1].copy(),
        "Middle third": df.iloc[i1:i2].copy(),
        "Late third": df.iloc[i2:].copy(),
    }

def build_html_report(context):
    html_parts = []
    css = """
    <style>
    body { font-family: Arial, Helvetica, sans-serif; margin: 0; padding: 0; color: #222; background: #fafafa; }
    .container { max-width: 1000px; margin: 0 auto; padding: 20px 16px 60px 16px; background: #fff; }
    h1 { font-size: 24px; margin: 0 0 10px 0; }
    h2 { font-size: 18px; margin-top: 24px; border-bottom: 1px solid #eee; padding-bottom: 6px; }
    h3 { font-size: 16px; margin-top: 18px; }
    p { line-height: 1.5; }
    .note { color: #555; font-size: 13px; }
    .figure { margin: 10px 0 4px 0; border: 1px solid #e6e6e6; padding: 6px; background: #fcfcfc; }
    .caption { font-size: 12px; color: #444; margin: 4px 2px; }
    .table { border-collapse: collapse; width: 100%; margin: 8px 0 14px 0; font-size: 14px; }
    .table th, .table td { border: 1px solid #ddd; padding: 6px 8px; text-align: right; }
    .table th:first-child, .table td:first-child { text-align: left; }
    .tag { display: inline-block; padding: 2px 6px; font-size: 12px; border-radius: 3px; background: #eef; margin-left: 6px; }
    .good { color: #0a7; font-weight: 600; }
    .bad { color: #c33; font-weight: 600; }
    .small { font-size: 12px; color: #666; }
    </style>
    """
    html_parts.append("<!DOCTYPE html><html><head><meta charset='utf-8'><title>NVIDIA Strategy Comparison Report</title>")
    html_parts.append(css)
    html_parts.append("</head><body><div class='container'>")

    html_parts.append("<h1>NVIDIA Strategy Comparison Report</h1>")
    html_parts.append("<p>This report analyzes three simple, long-only trading strategies on NVIDIA using the provided OHLCV data and adjusted prices. We compare performance across the full sample and across three non-overlapping chronological segments (early, middle, late thirds). We answer: which strategy grew a $1,000 initial investment the most and least per period; how risks differed via max drawdowns; and whether patterns are consistent. Signals are applied without forward-looking bias (positions act the next day), results use Adjusted Close, duplicate/missing dates are handled, and a simple per-trade cost is stated so results are net of costs.</p>")

    # Data notes
    html_parts.append("<h2>Data Handling and Assumptions</h2>")
    html_parts.append(f"<p>Data summary: {context['n_rows']} rows after cleaning; date range {context['date_min']} to {context['date_max']}.</p>")
    html_parts.append("<ul class='small'>")
    html_parts.append(f"<li>Duplicates by date removed: {context['n_dupes_removed']}; rows with missing Adjusted Close dropped: {context['n_missing_adj_close']}.</li>")
    html_parts.append("<li>Dates parsed and sorted chronologically; all computations are forward-looking-bias-free: positions for a day are based on signals from prior data and take effect the next trading day.</li>")
    html_parts.append("<li>Price basis: Adjusted Close used for returns, signals, and equity curves to account for corporate actions.</li>")
    html_parts.append(f"<li>Transaction costs: assumed {format_percent(context['transaction_cost'])} per position change (e.g., 0→1 or 1→0) on traded notional; all results are net of these costs.</li>")
    html_parts.append("<li>Strategy parameters were chosen solely based on segment length (number of observations) before backtesting and then held fixed within each segment; no future data were used to set parameters (no look-ahead).</li>")
    html_parts.append("</ul>")

    # Strategies
    html_parts.append("<h2>Strategies Evaluated</h2>")
    html_parts.append("<ul>")
    html_parts.append("<li>Buy-and-Hold baseline using Adjusted Close.</li>")
    html_parts.append(f"<li>Moving-average trend-following: long when short MA &gt; long MA; otherwise in cash. Full-sample windows around {context['ma_windows_full'][0]}/{context['ma_windows_full'][1]} (short/long); exact per-period parameters were fixed ex-ante and are listed below.</li>")
    html_parts.append(f"<li>Time-series momentum: long when lookback return &gt; 0, otherwise in cash. Full-sample lookback around {context['mom_lookback_full']} trading days; exact per-period parameters were fixed ex-ante and are listed below.</li>")
    html_parts.append("</ul>")

    # Full-sample performance summary table
    html_parts.append("<h2>Full-Sample Performance (Summary Table)</h2>")
    html_parts.append("<p class='small'>This table explicitly reports the final value and total return (%) for all three strategies on the full sample (net of costs).</p>")
    html_parts.append(context["tables_period"]["Full sample"].to_html(index=False, escape=False, classes='table', border=0))
    # Explicit overall full-sample Best/Worst statement
    html_parts.append(f"<p><b>Full-sample Best/Worst:</b> Best = <span class='good'>{context['full_best_name']}</span> at {format_currency(context['full_best_value'])}; "
                      f"Worst = <span class='bad'>{context['full_worst_name']}</span> at {format_currency(context['full_worst_value'])}.</p>")

    # Consolidated per-period performance table (numeric columns)
    html_parts.append("<h2>Per-Period Performance (All Periods, net of costs)</h2>")
    html_parts.append("<p class='small'>One row per (Period, Strategy) with two numeric columns. Total Return is expressed in percent (no % symbol). This table consolidates Full, Early, Middle, and Late thirds for auditability.</p>")
    html_parts.append(context["all_periods_table"].to_html(index=False, classes='table', border=0))

    # Figures
    html_parts.append("<h2>Equity Curves (Full Sample)</h2>")
    html_parts.append("<div class='figure'>")
    html_parts.append(context["fig_equity_svg"])
    html_parts.append("</div>")
    html_parts.append("<div class='caption'>Takeaway: The equity-curve comparison shows which strategy compounded the $1,000 fastest and how their growth paths differed over time.</div>")

    html_parts.append("<h2>Drawdowns (Full Sample)</h2>")
    html_parts.append("<div class='figure'>")
    html_parts.append(context["fig_drawdown_svg"])
    html_parts.append("</div>")
    html_parts.append("<div class='caption'>Takeaway: The drawdown plot highlights depth and duration of losses from peaks, indicating downside risk for each strategy.</div>")

    html_parts.append("<h2>Final Portfolio Value by Period</h2>")
    html_parts.append("<div class='figure'>")
    html_parts.append(context["fig_bars_svg"])
    html_parts.append("</div>")
    html_parts.append("<div class='caption'>Takeaway: The grouped bars show which strategy led in each period and by how much in final dollar value.</div>")

    # Best/worst identification by period (including full)
    html_parts.append("<h2>Best and Worst Strategies by Final Value</h2>")
    html_parts.append("<ul>")
    for period_name in context["periods_order"]:
        bw = context["best_worst"][period_name]
        html_parts.append(f"<li><b>{period_name}:</b> Best strategy by final value: <span class='good'>{bw['best_name']}</span> ({format_currency(bw['best_value'])}); Worst strategy: <span class='bad'>{bw['worst_name']}</span> ({format_currency(bw['worst_value'])}).</li>")
    html_parts.append("</ul>")
    html_parts.append("<p class='small note'>If multiple strategies tie on final value, they are shown as “Tie: A & B”. For narrative single-name references elsewhere, alphabetical order is used as a tie-break.</p>")

    # Period tables (all 4 periods, each with all 3 strategies)
    html_parts.append("<h2>Period Results: $1,000 initial investment (net of costs)</h2>")
    html_parts.append("<p class='small'>Each table below lists all three strategies (Buy & Hold, MA Trend, Time-Series Momentum) with final value and total return for the specified period.</p>")
    for period_name in context["periods_order"]:
        df_tbl = context["tables_period"][period_name]
        html_parts.append(f"<h3>{period_name}</h3>")
        html_parts.append(df_tbl.to_html(index=False, escape=False, classes='table', border=0))
    html_parts.append("<p class='small note'>Parameters were adapted to segment length before backtesting and then held fixed within each segment; no future data were used. In the event of identical final values (ties), ties are stated explicitly.</p>")

    # Full-sample risk metrics table (+ explicit bullet list)
    html_parts.append("<h2>Full-Sample Risk Metrics</h2>")
    html_parts.append(context["risk_table"].to_html(index=False, escape=False, classes='table', border=0))
    html_parts.append("<p class='small'>Sharpe ratios use daily strategy returns, a 0% risk-free rate, and an annualization factor of 252 trading days.</p>")
    html_parts.append("<ul>")
    for s in context["risk_bullets"]:
        html_parts.append(f"<li>{s}</li>")
    html_parts.append("</ul>")
    html_parts.append("<div class='caption'>Takeaway: Risk-adjusted performance (Sharpe) and drawdowns provide a perspective on trade-offs beyond final value.</div>")

    # Compact summary of best/worst with magnitude
    html_parts.append("<h2>Compact Best/Worst Summary (Magnitude of Outperformance)</h2>")
    html_parts.append(context["compact_bw_table"].to_html(index=False, escape=False, classes='table', border=0))
    html_parts.append("<p class='small'>Note: “pp” denotes percentage points.</p>")

    # Optional: parameter settings per period (for reproducibility)
    html_parts.append("<h2>Strategy Parameters Used per Period</h2>")
    html_parts.append(context["params_table"].to_html(index=False, escape=False, classes='table', border=0))

    # Insights
    html_parts.append("<h2>Data-Driven Insights</h2>")
    html_parts.append("<ol>")
    html_parts.append(f"<li>Full sample: {context['full_best_name']} ended at {format_currency(context['full_best_value'])} with Sharpe {context['full_best_sharpe']:.2f}, versus {context['full_worst_name']} at {format_currency(context['full_worst_value'])} and Sharpe {context['full_worst_sharpe']:.2f}, a final-value gap of {format_currency(context['full_best_value'] - context['full_worst_value'])}.</li>")
    html_parts.append(f"<li>Risk trade-off: {context['mdd_best_name']} contained max drawdown to {format_percent(context['mdd_best_value'])} vs. {format_percent(context['mdd_worst_value'])} for {context['mdd_worst_name']}, indicating materially better downside protection in the former.</li>")
    for s in context["insights_cross_period"]:
        html_parts.append(f"<li>{s}</li>")
    html_parts.append("</ol>")

    html_parts.append("</div></body></html>")
    return "".join(html_parts)

def main():
    parser = argparse.ArgumentParser(description="Compare simple trading strategies on NVIDIA using OHLCV CSV.")
    parser.add_argument("--data", required=True, help="Path to input CSV file with OHLCV data including Date and Adj Close.")
    args = parser.parse_args()

    try:
        if not os.path.exists(args.data):
            raise FileNotFoundError(f"File not found: {args.data}")

        df_raw = pd.read_csv(args.data)
        if df_raw.empty:
            raise ValueError("CSV file is empty.")

        # Identify columns
        date_col = find_col(df_raw, ["Date", "date"])
        adj_col = find_col(df_raw, ["Adj Close", "AdjClose", "Adj_Close", "Adjusted Close", "Adj. Close"])
        if date_col is None or adj_col is None:
            raise ValueError("Required columns not found. Expected at least 'Date' and 'Adj Close' (case-insensitive).")

        # Parse dates
        df_raw[date_col] = pd.to_datetime(df_raw[date_col], errors="coerce")

        # Drop rows with invalid dates
        n_invalid_dates = df_raw[date_col].isna().sum()
        if n_invalid_dates > 0:
            df_raw = df_raw.dropna(subset=[date_col])

        # Sort by date ascending
        df_raw = df_raw.sort_values(by=date_col)

        # Remove duplicate dates
        before = len(df_raw)
        df_raw = df_raw.drop_duplicates(subset=[date_col], keep="first")
        n_dupes_removed = before - len(df_raw)

        # Drop rows with missing Adj Close
        n_missing_adj_close = df_raw[adj_col].isna().sum()
        df = df_raw.dropna(subset=[adj_col]).copy()

        if df.empty or len(df) < 10:
            raise ValueError("Not enough valid data after cleaning to run the analysis (need at least 10 rows).")

        df = df[[date_col, adj_col]].rename(columns={date_col: "Date", adj_col: "AdjClose"})
        df = df.sort_values("Date").reset_index(drop=True)
        df.set_index("Date", inplace=True)

        date_min = df.index.min().date().isoformat()
        date_max = df.index.max().date().isoformat()
        n_rows = len(df)

        # Strategy parameterization (full sample defaults; segments auto-adapt, fixed ex-ante)
        sw_full, lw_full = get_ma_windows(n_rows)
        mom_full = get_mom_lookback(n_rows)

        # Transaction cost assumption
        transaction_cost = 0.0005  # 5 bps per position change

        # Prepare periods
        periods = split_periods(df)
        periods_order = ["Full sample", "Early third", "Middle third", "Late third"]

        # Strategy definitions (keys and human-readable names)
        strategies = [
            ("bh", "Buy & Hold"),
            ("ma", f"MA Trend (S/L={sw_full}/{lw_full})"),
            ("tsmom", f"Time-Series Momentum (L={mom_full}d)"),
        ]

        # Results containers
        tables_period = {}
        best_worst = {}
        final_values_for_bars = {"Buy & Hold": {}, "MA Trend": {}, "Time-Series Momentum": {}}
        period_returns = {"Full sample": {}, "Early third": {}, "Middle third": {}, "Late third": {}}
        params_used = {}

        # For full-sample figures and risk
        full_equities = {}
        full_drawdowns = {}
        full_returns = {}

        def summarize_best_worst(cands):
            # cands: list of (name, value)
            if not cands:
                return {
                    "best_name": "NA", "best_value": np.nan, "best_names": [],
                    "worst_name": "NA", "worst_value": np.nan, "worst_names": []
                }
            vals = [v for _, v in cands]
            maxv = max(vals)
            minv = min(vals)
            tol = 1e-9
            best_list = [n for n, v in cands if abs(v - maxv) <= tol]
            worst_list = [n for n, v in cands if abs(v - minv) <= tol]
            best_name = "Tie: " + " & ".join(sorted(best_list)) if len(best_list) > 1 else best_list[0]
            worst_name = "Tie: " + " & ".join(sorted(worst_list)) if len(worst_list) > 1 else worst_list[0]
            return {
                "best_name": best_name, "best_value": float(maxv), "best_names": best_list,
                "worst_name": worst_name, "worst_value": float(minv), "worst_names": worst_list
            }

        # Run each period and each strategy
        for period_name, dseg in periods.items():
            # Adjust parameters for segment length to ensure signals are valid (fixed ex-ante per segment)
            sw_seg, lw_seg = get_ma_windows(len(dseg))
            mom_seg = get_mom_lookback(len(dseg))
            params_used[period_name] = {
                "MA Short": sw_seg,
                "MA Long": lw_seg,
                "TSMom Lookback (d)": mom_seg
            }

            rows = []
            bw_candidates = []
            for key, label in strategies:
                if key == "bh":
                    params = {}
                    short_label = "Buy & Hold"
                elif key == "ma":
                    params = {"short_window": sw_seg, "long_window": lw_seg}
                    short_label = "MA Trend"
                else:
                    params = {"lookback": mom_seg}
                    short_label = "Time-Series Momentum"

                bt = backtest(dseg["AdjClose"], transaction_cost=transaction_cost, strategy=key, params=params, initial=1000.0)
                equity = bt["equity"]
                final_val = float(equity.iloc[-1])
                total_ret = final_val / 1000.0 - 1.0

                # Save rows for per-period table (formatted for readability)
                rows.append({
                    "Strategy": short_label,
                    "Final Value ($)": format_currency(final_val),
                    "Total Return (%)": format_percent(total_ret)
                })
                bw_candidates.append((short_label, final_val))

                # Save for full-sample risk and figures
                if period_name == "Full sample":
                    full_equities[short_label] = equity
                    full_returns[short_label] = bt["returns"]
                    full_drawdowns[short_label] = compute_drawdown(equity)

                # For consolidated table/bar chart and period return tracking
                final_values_for_bars[short_label][period_name] = final_val
                period_returns[period_name][short_label] = float(total_ret)

            # Build table for this period (ensures all 3 strategies are present)
            df_tbl = pd.DataFrame(rows)
            # Sort rows for consistent display
            order_map = {"Buy & Hold": 0, "MA Trend": 1, "Time-Series Momentum": 2}
            df_tbl["__order"] = df_tbl["Strategy"].map(order_map)
            df_tbl = df_tbl.sort_values("__order").drop(columns="__order")
            tables_period[period_name] = df_tbl

            # Best/worst for this period with tie handling
            bw_summary = summarize_best_worst(bw_candidates)
            best_worst[period_name] = bw_summary

        # Build consolidated all-periods performance table (numeric columns)
        all_rows = []
        strategy_order = ["Buy & Hold", "MA Trend", "Time-Series Momentum"]
        for period_name in periods_order:
            for strat in strategy_order:
                fv = final_values_for_bars[strat].get(period_name, np.nan)
                tr = period_returns[period_name].get(strat, np.nan)
                all_rows.append({
                    "Period": period_name,
                    "Strategy": strat,
                    "Final Value ($)": round(fv, 2) if np.isfinite(fv) else np.nan,
                    "Total Return (%)": round(tr * 100.0, 2) if np.isfinite(tr) else np.nan
                })
        all_periods_table = pd.DataFrame(all_rows)
        # Enforce ordering
        all_periods_table["__porder"] = all_periods_table["Period"].map({p: i for i, p in enumerate(periods_order)})
        all_periods_table["__sorder"] = all_periods_table["Strategy"].map({s: i for i, s in enumerate(strategy_order)})
        all_periods_table = all_periods_table.sort_values(["__porder", "__sorder"]).drop(columns=["__porder", "__sorder"])

        # Full-sample risk metrics (include all three strategies)
        risk_metrics = []
        for strat_name in ["Buy & Hold", "MA Trend", "Time-Series Momentum"]:
            eq = full_equities.get(strat_name)
            rets = full_returns.get(strat_name)
            if eq is None or rets is None:
                mdd = np.nan
                sharpe = np.nan
            else:
                dd = compute_drawdown(eq)
                mdd = float(dd.min()) if len(dd) > 0 else np.nan
                sharpe = float(annualized_sharpe(rets))
            risk_metrics.append({
                "Strategy": strat_name,
                "Max Drawdown (%)": format_percent(mdd if not np.isnan(mdd) else np.nan),
                "Annualized Sharpe (0% RF)": f"{sharpe:.2f}" if not np.isnan(sharpe) else "NA"
            })
        risk_table = pd.DataFrame(risk_metrics)

        # Explicit risk bullets
        risk_bullets = []
        for row in risk_metrics:
            risk_bullets.append(f"{row['Strategy']}: Max Drawdown {row['Max Drawdown (%)']}, Sharpe {row['Annualized Sharpe (0% RF)']}")

        # Determine full-sample best/worst and sharpe info (tie-break alphabetically for narrative)
        full_bws = best_worst["Full sample"]
        def first_alpha(lst):
            return sorted(lst)[0] if lst else "NA"
        full_best_name = full_bws["best_name"]
        full_best_value = full_bws["best_value"]
        full_worst_name = full_bws["worst_name"]
        full_worst_value = full_bws["worst_value"]
        if isinstance(full_best_name, str) and full_best_name.startswith("Tie:"):
            full_best_name_display = first_alpha(full_bws.get("best_names", []))
        else:
            full_best_name_display = full_best_name
        if isinstance(full_worst_name, str) and full_worst_name.startswith("Tie:"):
            full_worst_name_display = first_alpha(full_bws.get("worst_names", []))
        else:
            full_worst_name_display = full_worst_name

        # Sharpe lookup
        sharpe_lookup = {row["Strategy"]: row["Annualized Sharpe (0% RF)"] for _, row in risk_table.iterrows()}
        def parse_sharpe(s):
            try:
                return float(s)
            except Exception:
                return np.nan
        full_best_sharpe = parse_sharpe(sharpe_lookup.get(full_best_name_display, "NA"))
        full_worst_sharpe = parse_sharpe(sharpe_lookup.get(full_worst_name_display, "NA"))

        # MDD extremes
        mdd_values = []
        for strat_name in ["Buy & Hold", "MA Trend", "Time-Series Momentum"]:
            eq = full_equities.get(strat_name)
            if eq is not None:
                mdd = float(compute_drawdown(eq).min())
            else:
                mdd = np.nan
            mdd_values.append((strat_name, mdd))
        mdd_values_clean = [(n, v) for n, v in mdd_values if not np.isnan(v)]
        if mdd_values_clean:
            # most negative is worst; least negative (closest to 0) is best
            mdd_worst_name, mdd_worst_value = sorted(mdd_values_clean, key=lambda x: x[1])[0]
            mdd_best_name, mdd_best_value = sorted(mdd_values_clean, key=lambda x: x[1])[-1]
        else:
            mdd_worst_name = mdd_best_name = "NA"
            mdd_worst_value = mdd_best_value = np.nan

        # Build figures (small size, low DPI)
        sns.set_theme(style="whitegrid")

        # Equity curves figure
        fig1, ax1 = plt.subplots(figsize=(7.5, 5), dpi=80)
        colors = {"Buy & Hold": "#1f77b4", "MA Trend": "#2ca02c", "Time-Series Momentum": "#d62728"}
        for strat_name in ["Buy & Hold", "MA Trend", "Time-Series Momentum"]:
            eq = full_equities.get(strat_name)
            if eq is not None:
                ax1.plot(eq.index, eq.values, label=strat_name, linewidth=1.2, color=colors.get(strat_name, None))
        ax1.set_title("Equity Curves (Full Sample), $1,000 start (net of costs)", fontsize=12)
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Equity ($)")
        ax1.legend(fontsize=9, frameon=False)
        fig_equity_svg = svg_from_fig(fig1, dpi=80)

        # Drawdowns figure
        fig2, ax2 = plt.subplots(figsize=(7.5, 4.5), dpi=80)
        for strat_name in ["Buy & Hold", "MA Trend", "Time-Series Momentum"]:
            eq = full_equities.get(strat_name)
            if eq is not None:
                dd = compute_drawdown(eq)
                ax2.plot(dd.index, dd.values * 100.0, label=strat_name, linewidth=1.0, color=colors.get(strat_name, None))
        ax2.axhline(0, color="#999", linewidth=0.8)
        ax2.set_title("Drawdown (Full Sample)", fontsize=12)
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Drawdown (%)")
        ax2.legend(fontsize=9, frameon=False)
        fig_drawdown_svg = svg_from_fig(fig2, dpi=80)

        # Bar chart of final values per period per strategy
        bars_rows = []
        for strat_key, strat_label in [("Buy & Hold", "Buy & Hold"), ("MA Trend", "MA Trend"), ("Time-Series Momentum", "Time-Series Momentum")]:
            for period_name in periods_order:
                v = final_values_for_bars[strat_key].get(period_name, np.nan)
                bars_rows.append({"Strategy": strat_label, "Period": period_name, "FinalValue": v})
        bars_df = pd.DataFrame(bars_rows)
        fig3, ax3 = plt.subplots(figsize=(7.5, 5.0), dpi=80)
        x = np.arange(len(periods_order))
        width = 0.26
        for i, strat_name in enumerate(["Buy & Hold", "MA Trend", "Time-Series Momentum"]):
            vals = [bars_df[(bars_df["Strategy"] == strat_name) & (bars_df["Period"] == p)]["FinalValue"].values
                    for p in periods_order]
            vals = [v[0] if len(v) > 0 else np.nan for v in vals]
            ax3.bar(x + (i - 1) * width, vals, width, label=strat_name, color=colors.get(strat_name, None))
        ax3.set_xticks(x)
        ax3.set_xticklabels(periods_order, rotation=0)
        ax3.set_ylabel("Final Value ($)")
        ax3.set_title("Final Portfolio Value by Period (net of costs)", fontsize=12)
        ax3.legend(fontsize=9, frameon=False)
        fig_bars_svg = svg_from_fig(fig3, dpi=80)

        # Period win summary for caption (use full label including tie if any)
        wins = []
        for period_name in periods_order:
            bw = best_worst[period_name]
            wins.append(f"{period_name}: {bw['best_name']}")
        period_win_summary = "; ".join(wins)

        # Build cross-period insights
        def safe_fmt_pct(x):
            try:
                return f"{x*100:.1f}%"
            except Exception:
                return "NA"

        insights_cross_period = []
        def third_insight(pname, short_label):
            try:
                bw = best_worst[pname]
                best_label = bw["best_name"]
                worst_label = bw["worst_name"]
                ev_bh = final_values_for_bars["Buy & Hold"][pname]
                ev_ma = final_values_for_bars["MA Trend"][pname]
                ev_ts = final_values_for_bars["Time-Series Momentum"][pname]
                rt_bh = period_returns[pname]["Buy & Hold"]
                rt_ma = period_returns[pname]["MA Trend"]
                rt_ts = period_returns[pname]["Time-Series Momentum"]
                best_val = bw["best_value"]
                worst_val = bw["worst_value"]
                diff_val = best_val - worst_val
                return (
                    f"{short_label}: Buy & Hold {format_currency(ev_bh)} ({safe_fmt_pct(rt_bh)}), "
                    f"MA Trend {format_currency(ev_ma)} ({safe_fmt_pct(rt_ma)}), "
                    f"Time-Series Momentum {format_currency(ev_ts)} ({safe_fmt_pct(rt_ts)}); "
                    f"Best = {best_label}, Worst = {worst_label}; best led worst by {format_currency(diff_val)}."
                )
            except Exception:
                return None

        ins_early = third_insight("Early third", "Early third")
        ins_middle = third_insight("Middle third", "Middle third")
        ins_late = third_insight("Late third", "Late third")
        for s in [ins_early, ins_middle, ins_late]:
            if s:
                insights_cross_period.append(s)

        # Compact best/worst summary with magnitudes (handles ties explicitly)
        compact_rows = []
        for period_name in periods_order:
            bw = best_worst[period_name]
            best_val = bw["best_value"]
            worst_val = bw["worst_value"]
            diff_val = best_val - worst_val if np.isfinite(best_val) and np.isfinite(worst_val) else np.nan

            best_label = bw["best_name"]
            worst_label = bw["worst_name"]

            if isinstance(best_label, str) and not best_label.startswith("Tie:") and isinstance(worst_label, str) and not worst_label.startswith("Tie:"):
                best_rt = period_returns[period_name].get(best_label, np.nan)
                worst_rt = period_returns[period_name].get(worst_label, np.nan)
                diff_pp = (best_rt - worst_rt) * 100.0 if not (np.isnan(best_rt) or np.isnan(worst_rt)) else np.nan
            else:
                diff_pp = 0.0 if (np.isfinite(best_val) and np.isfinite(worst_val)) else np.nan

            compact_rows.append({
                "Period": period_name,
                "Best": best_label,
                "Best Final ($)": format_currency(best_val) if np.isfinite(best_val) else "NA",
                "Worst": worst_label,
                "Worst Final ($)": format_currency(worst_val) if np.isfinite(worst_val) else "NA",
                "$ Outperformance": format_currency(diff_val) if np.isfinite(diff_val) else "NA",
                "%pt Outperformance": f"{diff_pp:.1f}pp" if (isinstance(diff_pp, float) and np.isfinite(diff_pp)) else "NA"
            })
        compact_bw_table = pd.DataFrame(compact_rows)

        # Parameters table for reproducibility
        params_rows = []
        for p in periods_order:
            pr = params_used[p]
            params_rows.append({
                "Period": p,
                "MA Short": pr["MA Short"],
                "MA Long": pr["MA Long"],
                "TSMom Lookback (d)": pr["TSMom Lookback (d)"]
            })
        params_table = pd.DataFrame(params_rows)

        # Build context
        context = {
            "n_rows": n_rows,
            "date_min": date_min,
            "date_max": date_max,
            "n_dupes_removed": n_dupes_removed,
            "n_missing_adj_close": n_missing_adj_close,
            "transaction_cost": transaction_cost,
            "ma_windows_full": (sw_full, lw_full),
            "mom_lookback_full": mom_full,
            "fig_equity_svg": fig_equity_svg,
            "fig_drawdown_svg": fig_drawdown_svg,
            "fig_bars_svg": fig_bars_svg,
            "tables_period": tables_period,
            "all_periods_table": all_periods_table,
            "risk_table": risk_table,
            "risk_bullets": risk_bullets,
            "periods_order": periods_order,
            "best_worst": best_worst,
            "full_best_name": full_best_name_display,
            "full_best_value": full_best_value,
            "full_worst_name": full_worst_name_display,
            "full_worst_value": full_worst_value,
            "full_best_sharpe": full_best_sharpe if not np.isnan(full_best_sharpe) else float('nan'),
            "full_worst_sharpe": full_worst_sharpe if not np.isnan(full_worst_sharpe) else float('nan'),
            "mdd_worst_name": mdd_worst_name,
            "mdd_worst_value": mdd_worst_value,
            "mdd_best_name": mdd_best_name,
            "mdd_best_value": mdd_best_value,
            "period_win_summary": period_win_summary,
            "insights_cross_period": insights_cross_period,
            "compact_bw_table": compact_bw_table,
            "params_table": params_table,
        }

        html = build_html_report(context)
        sys.stdout.write(html)

    except Exception as e:
        # Output a minimal HTML with the error message for robustness
        err_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Error</title>
<style>
body {{ font-family: Arial, Helvetica, sans-serif; background: #fafafa; }}
.container {{ max-width: 900px; margin: 40px auto; background: #fff; padding: 20px; border: 1px solid #eee; }}
pre {{ background: #f7f7f7; padding: 12px; overflow: auto; border: 1px solid #eee; }}
</style>
</head><body><div class="container">
<h1>Processing Error</h1>
<p>There was an error while generating the report. Please check the data file and try again.</p>
<p><b>Message:</b> {str(e)}</p>
<details><summary>Traceback</summary><pre>{traceback.format_exc()}</pre></details>
</div></body></html>"""
        sys.stdout.write(err_html)

if __name__ == "__main__":
    main()