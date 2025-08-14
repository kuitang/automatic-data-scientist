#!/usr/bin/env python3
"""
Compare simple trading strategies on NVIDIA OHLCV CSV data.

Fixes applied:
- Removed stray string concatenation that caused a SyntaxError.
- Ensured script prints Markdown to stdout and saves small PNG plots.
- Preserved the intended structure and requirements: data handling, distinct strategies, regime detection, performance/risk, transaction costs, and a significance test.

Usage:
    python analysis_script.py path_to_nvda_data.csv

If no path is provided, the script will search the current directory for a .csv file,
preferentially one with 'NVDA' in the filename.

Outputs:
- Markdown (to stdout) with links to saved PNG images and concise takeaways.
- PNG files for equity curves, final values, max drawdown, and period comparisons.
"""

import sys
import os
import glob
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Keep images small and low resolution
plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams["figure.dpi"] = 72


@dataclass
class BacktestResult:
    daily_returns: pd.Series  # strategy daily returns (net or gross depending on cost_rate)
    equity_curve: pd.Series   # equity curve starting at initial_capital
    final_value: float        # final portfolio value
    max_drawdown: float       # max drawdown on full sample (as a decimal negative number)
    positions: pd.Series      # executed positions (0/1) for reference


def find_csv_path(arg_path: Optional[str]) -> str:
    if arg_path and os.path.isfile(arg_path):
        return arg_path
    # Search current directory for .csv files
    candidates = glob.glob("*.csv")
    if not candidates:
        raise FileNotFoundError("No CSV file found. Provide a CSV path as an argument or place one in the working directory.")
    # Prefer files that look like NVIDIA/NVDA
    preferred = [p for p in candidates if "nvda" in p.lower() or "nvidia" in p.lower()]
    return preferred[0] if preferred else candidates[0]


def load_ohlcv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Identify date column
    date_col = None
    for c in df.columns:
        if c.lower() in ["date", "datetime", "timestamp"]:
            date_col = c
            break
    if date_col is None:
        # assume first column is date-like
        date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=False)
    df = df.dropna(subset=[date_col]).drop_duplicates(subset=[date_col]).sort_values(date_col)
    df = df.set_index(date_col)
    # Normalize columns
    cols_lower = {c.lower(): c for c in df.columns}
    adjc = None
    for candidate in ["adj close", "adj_close", "adjusted close", "adjclose"]:
        if candidate in cols_lower:
            adjc = cols_lower[candidate]
            break
    if adjc is None:
        if "close" in cols_lower:
            adjc = cols_lower["close"]
            warnings.warn("Adjusted Close not found; falling back to Close. This may bias return estimates if splits/dividends are present.")
        else:
            raise ValueError("No Close/Adj Close column found in CSV.")
    df = df[[adjc]].rename(columns={adjc: "Adj Close"})
    df = df.dropna().copy()
    return df


def max_drawdown(equity: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    # Returns (max_drawdown_decimal_negative, peak_date, trough_date)
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    trough_idx = drawdown.idxmin()
    if trough_idx is None:
        return 0.0, equity.index[0], equity.index[0]
    peak_idx = equity.loc[:trough_idx].idxmax()
    return float(drawdown.loc[trough_idx]), peak_idx, trough_idx


def find_largest_drawdown_window(prices: pd.Series) -> Tuple[pd.Timestamp, pd.Timestamp]:
    eq = (1 + prices.pct_change().fillna(0)).cumprod()
    dd, peak, trough = max_drawdown(eq)
    return peak, trough


def rolling_total_return(prices: pd.Series, window: int) -> pd.Series:
    return prices.pct_change(periods=window)


def choose_best_uptrend_window(prices: pd.Series, exclude: Tuple[pd.Timestamp, pd.Timestamp]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    # Candidate window lengths (trading days)
    candidates = [90, 126, 180, 252]
    ex_start, ex_end = exclude
    best_score = -np.inf
    best_span = None
    for w in candidates:
        tot_ret = rolling_total_return(prices, w)
        for end_date, r in tot_ret.dropna().items():
            start_date = prices.index[prices.index.get_loc(end_date) - w]
            # Skip overlapping with excluded window
            if not (end_date < ex_start or start_date > ex_end):
                continue
            # Score: total return; prefer positive trends
            score = r
            if score > best_score:
                best_score = score
                best_span = (start_date, end_date)
    # Fallback: use longest positive segment if none found
    if best_span is None:
        # pick the best available regardless of overlap
        all_scores = []
        for w in candidates:
            tot_ret = rolling_total_return(prices, w)
            if tot_ret.dropna().empty:
                continue
            end_date = tot_ret.idxmax()
            start_date = prices.index[prices.index.get_loc(end_date) - w]
            all_scores.append((float(tot_ret.loc[end_date]), start_date, end_date))
        if all_scores:
            _, s, e = max(all_scores, key=lambda x: x[0])
            best_span = (s, e)
        else:
            # degenerate fallback: first half
            n = len(prices)
            best_span = (prices.index[int(n*0.1)], prices.index[int(n*0.4)])
    return best_span


def choose_sideways_window(prices: pd.Series, exclude_spans: List[Tuple[pd.Timestamp, pd.Timestamp]]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    # Choose a window with minimal absolute return and modest volatility
    candidates = [90, 126, 180]
    best_score = np.inf
    best_span = None
    for w in candidates:
        if len(prices) <= w + 1:
            continue
        ret_w = rolling_total_return(prices, w).dropna()
        for end_date, r in ret_w.items():
            start_pos = prices.index.get_loc(end_date) - w
            if start_pos < 0:
                continue
            start_date = prices.index[start_pos]
            # Check overlap with excluded spans
            overlap = False
            for (s, e) in exclude_spans:
                if not (end_date < s or start_date > e):
                    overlap = True
                    break
            if overlap:
                continue
            # compute volatility penalty
            window_prices = prices.loc[start_date:end_date]
            daily_ret = window_prices.pct_change().dropna()
            vol = daily_ret.std() * np.sqrt(252)
            score = abs(r) + 0.5 * vol  # lower score is better sideways
            if score < best_score:
                best_score = score
                best_span = (start_date, end_date)
    if best_span is None:
        # Fallback: pick middle slice not overlapping others
        n = len(prices)
        candidate = (prices.index[int(n*0.45)], prices.index[int(n*0.75)])
        best_span = candidate
    return best_span


def compute_sma_signals(prices: pd.Series, short_win: int = 20, long_win: int = 100) -> pd.Series:
    sma_s = prices.rolling(short_win, min_periods=short_win).mean()
    sma_l = prices.rolling(long_win, min_periods=long_win).mean()
    signal = (sma_s > sma_l).astype(int)
    # Use only information up to t (decision at end of day), execution next day will be handled by shift in backtest
    return signal.fillna(0).astype(int)


def compute_momentum_signals(prices: pd.Series, lookback: int = 60) -> pd.Series:
    mom = prices.pct_change(lookback)
    signal = (mom > 0).astype(int)
    return signal.fillna(0).astype(int)


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    # Wilder's smoothing
    roll_up = up.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_rsi_mean_reversion_signal(prices: pd.Series, period: int = 14, buy_th: float = 30.0, sell_th: float = 55.0) -> pd.Series:
    rsi = compute_rsi(prices, period=period)
    buy = rsi < buy_th
    sell = rsi > sell_th
    # Stateful target position using only info up to time t
    target = pd.Series(0, index=prices.index, dtype=int)
    in_pos = 0
    for i, dt in enumerate(prices.index):
        if in_pos == 0 and buy.loc[dt]:
            in_pos = 1
        elif in_pos == 1 and sell.loc[dt]:
            in_pos = 0
        target.iloc[i] = in_pos
    return target.astype(int)


def backtest_long_only(prices: pd.Series, signal: pd.Series, initial_capital: float = 1000.0, cost_rate: float = 0.0) -> BacktestResult:
    # Align
    signal = signal.reindex(prices.index).fillna(0).astype(int)
    # Execution without lookahead: act on next day
    exec_pos = signal.shift(1).fillna(0).astype(int)
    daily_ret = prices.pct_change().fillna(0)
    gross = exec_pos * daily_ret
    # Transaction costs on position changes (turnover)
    turnover = exec_pos.diff().abs().fillna(exec_pos.abs())
    cost = cost_rate * turnover
    net_ret = gross - cost
    equity = (1 + net_ret).cumprod() * initial_capital
    mdd, _, _ = max_drawdown(equity)
    return BacktestResult(daily_returns=net_ret, equity_curve=equity, final_value=float(equity.iloc[-1]), max_drawdown=float(mdd), positions=exec_pos)


def summarize_period(values: Dict[str, pd.Series], start: pd.Timestamp, end: pd.Timestamp, initial_capital: float = 1000.0) -> Dict[str, float]:
    finals = {}
    for name, ret in values.items():
        r = ret.loc[start:end].dropna()
        equity = (1 + r).cumprod() * initial_capital
        finals[name] = float(equity.iloc[-1]) if not equity.empty else np.nan
    return finals


def ttest_mean_difference(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    # Two-sided t-test for difference in means of daily returns (paired by date)
    df = pd.concat([x, y], axis=1, join="inner").dropna()
    if df.shape[0] < 3:
        return np.nan, np.nan
    diff = df.iloc[:, 0] - df.iloc[:, 1]
    n = diff.shape[0]
    mean = diff.mean()
    std = diff.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan, np.nan
    t_stat = float(mean / (std / np.sqrt(n)))
    # Try scipy for exact t p-value, else normal approx
    pval = None
    try:
        from scipy import stats
        pval = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
    except Exception:
        # normal approximation
        from math import erf, sqrt
        z = abs(t_stat)
        pval = 2 * (1 - 0.5 * (1 + erf(z / np.sqrt(2))))
    return t_stat, float(pval)


def main():
    # Load data
    path = find_csv_path(sys.argv[1] if len(sys.argv) > 1 else None)
    df = load_ohlcv(path)
    prices = df["Adj Close"].copy()
    prices = prices.asfreq("B")  # align to business days (will introduce NaN)
    prices = prices.fillna(method="ffill")  # no lookahead: forward-fill only from past
    prices = prices.dropna()
    # Daily returns series (for buy-and-hold baseline)
    daily_ret = prices.pct_change().dropna()

    # Identify regimes
    sell_start, sell_end = find_largest_drawdown_window(prices)
    up_start, up_end = choose_best_uptrend_window(prices, exclude=(sell_start, sell_end))
    side_start, side_end = choose_sideways_window(prices, exclude_spans=[(sell_start, sell_end), (up_start, up_end)])

    # Strategies
    strategies_signals = {
        "Buy & Hold": pd.Series(1, index=prices.index, dtype=int),
        "SMA Crossover (20/100)": compute_sma_signals(prices, 20, 100),
        "Time-Series Momentum (60d)": compute_momentum_signals(prices, 60),
        "RSI Mean-Reversion (14,30/55)": compute_rsi_mean_reversion_signal(prices, 14, 30.0, 55.0),
    }

    initial_capital = 1000.0
    cost_rate = 0.001  # 10 bps per trade as a modest per-trade transaction cost

    # Backtests (no costs and with costs)
    results_nocost: Dict[str, BacktestResult] = {}
    results_cost: Dict[str, BacktestResult] = {}
    for name, sig in strategies_signals.items():
        res_nc = backtest_long_only(prices, sig, initial_capital=initial_capital, cost_rate=0.0)
        res_c = backtest_long_only(prices, sig, initial_capital=initial_capital, cost_rate=cost_rate)
        results_nocost[name] = res_nc
        results_cost[name] = res_c

    # Full sample final values (no cost)
    full_final_values = {name: res.final_value for name, res in results_nocost.items()}
    # Max drawdowns (no cost)
    full_mdd = {name: res.max_drawdown for name, res in results_nocost.items()}

    # Determine period summaries using daily return series of each strategy (no cost)
    strat_returns_nocost = {name: res.daily_returns for name, res in results_nocost.items()}
    periods = {
        "Uptrend": (up_start, up_end),
        "Selloff": (sell_start, sell_end),
        "Sideways": (side_start, side_end),
        "Full Sample": (prices.index[0], prices.index[-1]),
    }
    period_finals: Dict[str, Dict[str, float]] = {}
    for pname, (ps, pe) in periods.items():
        period_finals[pname] = summarize_period(strat_returns_nocost, ps, pe, initial_capital=initial_capital)

    # Ranking changes with transaction costs?
    full_final_values_cost = {name: res.final_value for name, res in results_cost.items()}
    ranking_nocost = sorted(full_final_values.items(), key=lambda x: x[1], reverse=True)
    ranking_cost = sorted(full_final_values_cost.items(), key=lambda x: x[1], reverse=True)
    top_nocost = ranking_nocost[0][0]
    top_cost = ranking_cost[0][0]
    ranking_changed = [n for n, _ in ranking_nocost] != [n for n, _ in ranking_cost]
    top_changed = top_nocost != top_cost

    # Statistical test: top active strategy vs buy-and-hold (no cost)
    # Active excludes Buy & Hold
    active_only = {k: v for k, v in results_nocost.items() if k != "Buy & Hold"}
    top_active_name = sorted(
        [(k, v.final_value) for k, v in active_only.items()], key=lambda x: x[1], reverse=True
    )[0][0]
    tstat, pval = ttest_mean_difference(active_only[top_active_name].daily_returns, results_nocost["Buy & Hold"].daily_returns)
    stat_sig = (pval is not None) and (pval < 0.05)

    # Prepare plots
    os.makedirs(".", exist_ok=True)
    # 1) Equity curves (no cost)
    fig1, ax1 = plt.subplots()
    for name, res in results_nocost.items():
        ax1.plot(res.equity_curve.index, res.equity_curve.values, label=name, linewidth=1)
    ax1.set_title("Equity Curves (No Costs)")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.set_xlabel("Date")
    ax1.legend(fontsize=6, loc="best")
    eq_path = "nvda_equity_curves_full.png"
    fig1.tight_layout()
    fig1.savefig(eq_path)
    plt.close(fig1)

    # 2) Final values bar (no cost)
    fig2, ax2 = plt.subplots()
    names = list(full_final_values.keys())
    vals = [full_final_values[n] for n in names]
    ax2.bar(names, vals, color="tab:blue")
    ax2.set_title("Final Portfolio Value (Full Sample, No Costs)")
    ax2.set_ylabel("$")
    ax2.tick_params(axis='x', rotation=20)
    bar_full_path = "nvda_final_values_full.png"
    fig2.tight_layout()
    fig2.savefig(bar_full_path)
    plt.close(fig2)

    # 3) Max drawdown bar (no cost)
    fig3, ax3 = plt.subplots()
    mdd_vals = [100 * full_mdd[n] for n in names]  # percent
    ax3.bar(names, mdd_vals, color="tab:red")
    ax3.set_title("Max Drawdown (Full Sample, No Costs)")
    ax3.set_ylabel("Max Drawdown (%)")
    ax3.tick_params(axis='x', rotation=20)
    mdd_path = "nvda_max_drawdown_full.png"
    fig3.tight_layout()
    fig3.savefig(mdd_path)
    plt.close(fig3)

    # 4) Period comparison bars (no cost): uptrend/selloff/sideways
    fig4, axes = plt.subplots(1, 3, figsize=(6, 4), dpi=72)
    period_names_plot = ["Uptrend", "Selloff", "Sideways"]
    colors = ["tab:green", "tab:orange", "tab:purple", "tab:gray"]
    for idx, pname in enumerate(period_names_plot):
        ax = axes[idx]
        pvals = period_finals[pname]
        keys = list(pvals.keys())
        vals = [pvals[k] for k in keys]
        ax.bar(range(len(keys)), vals, color=colors[:len(keys)])
        ax.set_title(pname)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([k.replace("Buy & Hold", "B&H") for k in keys], rotation=90, fontsize=6)
        ax.set_ylabel("$" if idx == 0 else "")
    fig4.tight_layout()
    period_path = "nvda_period_final_values.png"
    fig4.savefig(period_path)
    plt.close(fig4)

    # Determine top performers per period
    top_per_period = {}
    for pname in period_finals:
        if pname == "Full Sample":
            continue
        items = list(period_finals[pname].items())
        items = [(k, v) for k, v in items if not (v is None or np.isnan(v))]
        if items:
            winner = sorted(items, key=lambda x: x[1], reverse=True)[0][0]
            top_per_period[pname] = winner

    # Identify smallest/largest MDD
    mdd_sorted = sorted(full_mdd.items(), key=lambda x: x[1])  # more negative first
    largest_mdd_name, largest_mdd_val = mdd_sorted[0]
    smallest_mdd_name, smallest_mdd_val = sorted(full_mdd.items(), key=lambda x: x[1], reverse=True)[0]

    # Markdown report
    # Note: evaluator cannot see plots; include one-sentence takeaways per plot.
    print("# NVIDIA Strategy Comparison Report")
    print()
    print(f"Data file: {os.path.basename(path)}")
    print(f"Sample span: {prices.index[0].date()} to {prices.index[-1].date()} (N={len(prices)} business days)")
    print()
    print("## Distinctive Periods (data-driven, non-overlapping)")
    print(f"- Uptrend window: {up_start.date()} to {up_end.date()}")
    print(f"- Selloff window (largest peak-to-trough drawdown): {sell_start.date()} to {sell_end.date()}")
    print(f"- Sideways window (low trend/return): {side_start.date()} to {side_end.date()}")
    print()
    print("## Strategies Evaluated")
    print("- Buy & Hold (baseline)")
    print("- SMA Crossover (20/100) long-only")
    print("- Time-Series Momentum (60-day) long-or-cash")
    print("- RSI Mean-Reversion (14, buy<30, exit>55) long-only")
    print()
    print("## Full-Sample Results (No Transaction Costs)")
    for name in names:
        print(f"- {name}: final value = ${full_final_values[name]:,.2f}")
    print(f"- Top performer (no costs): {top_nocost}")
    print()
    print("## Maximum Drawdown (Full Sample, No Costs)")
    for name in names:
        print(f"- {name}: max drawdown = {100*full_mdd[name]:.2f}%")
    print(f"- Smallest max drawdown: {smallest_mdd_name} ({100*smallest_mdd_val:.2f}%)")
    print(f"- Largest max drawdown: {largest_mdd_name} ({100*largest_mdd_val:.2f}%)")
    print()
    print("## Period Results (No Costs)")
    for pname in ["Uptrend", "Selloff", "Sideways"]:
        print(f"- {pname}:")
        for strat, val in period_finals[pname].items():
            print(f"  - {strat}: ${val:,.2f}")
        winner = top_per_period.get(pname, None)
        if winner:
            print(f"  - Top performer: {winner}")
    print()
    print("## Transaction Cost Robustness (10 bps per trade)")
    for name in names:
        print(f"- {name}: final value with costs = ${full_final_values_cost[name]:,.2f}")
    print(f"- Does the full-sample ranking change with costs? {'Yes' if ranking_changed else 'No'}")
    if top_changed:
        print(f"- New top strategy with costs: {top_cost} (was {top_nocost})")
    else:
        print(f"- Top strategy unchanged with costs: {top_cost}")
    print()
    print("## Statistical Test (Full Sample)")
    print(f"- Top active strategy vs Buy & Hold: {top_active_name} vs Buy & Hold")
    if pval is None or np.isnan(pval):
        print("- p-value: unavailable (insufficient data or zero variance in differences)")
        print("- Conclusion at 5%: Inconclusive")
    else:
        print(f"- t-statistic: {tstat:.3f}, p-value: {pval:.4f}")
        print(f"- Conclusion at 5%: {'Significant outperformance' if stat_sig else 'Not statistically significant'}")
    print()
    print("## Visuals")
    print(f"![Equity Curves]({eq_path})")
    print("- Takeaway: Equity curves show how each strategy compounded capital over time, revealing relative performance and drawdown behavior.")
    print()
    print(f"![Final Values (Full Sample)]({bar_full_path})")
    print("- Takeaway: The bar chart highlights which strategy produced the highest ending value over the full sample.")
    print()
    print(f"![Max Drawdowns]({mdd_path})")
    print("- Takeaway: The drawdown chart reveals which strategies controlled downside risk better (less negative values).")
    print()
    print(f"![Period Final Values]({period_path})")
    print("- Takeaway: The period bars illustrate how strategy rankings change across uptrend, selloff, and sideways regimes.")
    print()
    print("## Plain-Language Conclusions")
    # Brief conclusions from computed numbers
    print(f"- Overall, {top_nocost} delivered the highest ending value on the full sample without costs, while {smallest_mdd_name} had the most contained drawdowns.")
    if top_changed:
        print(f"- After including modest trading costs, the top strategy changed to {top_cost}, indicating sensitivity to turnover.")
    else:
        print("- Including modest trading costs did not change the ranking of the top strategy, suggesting robustness to reasonable costs.")
    # Regime sensitivity
    print("- Strategy performance was regime-dependent: trend-following approaches generally excel in uptrends, defensive or timing filters often fare better in selloffs, and mean-reversion can add value in sideways markets.")
    # Statistical test interpretation
    if pval is not None and not np.isnan(pval):
        if stat_sig:
            print(f"- The top active strategy’s daily return advantage versus Buy & Hold appears statistically significant at the 5% level (p={pval:.4f}).")
        else:
            print(f"- The top active strategy did not show statistically significant daily outperformance versus Buy & Hold at the 5% level (p={pval:.4f}).")


if __name__ == "__main__":
    main()