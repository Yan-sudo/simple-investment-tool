"""
量化交易可视化平台 — Quantitative Trading Visual Dashboard
=========================================================

Launch:
    streamlit run gui/app.py

Features:
    - Stock & Option strategy backtest visualization
    - Interactive Plotly equity curves with drawdown overlay
    - Trade log table with filtering
    - Option chain explorer
    - Alpaca API integration (real data + paper trading)
    - Full dark theme
"""

import sys
import os
import math

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from data.market_data import generate_synthetic_data, generate_regime_data, MarketData
from engine.backtest import BacktestEngine, BacktestResult
from evaluation.metrics import calculate_all_metrics, rate_strategy
from strategies.ma_crossover import MACrossoverStrategy
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.dual_thrust import DualThrustStrategy

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="量化交易平台 | Quant Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for polished dark look ────────────────────────────────────────

st.markdown("""
<style>
    /* KPI metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1A1F2E 0%, #252B3B 100%);
        border: 1px solid #2D3548;
        border-radius: 10px;
        padding: 15px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] label {
        color: #8892A4 !important;
        font-size: 0.85rem !important;
    }
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        border-right: 1px solid #2D3548;
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
    }
    /* Table styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        color: #00D4AA;
        border-bottom: 1px solid #2D3548;
        padding-bottom: 6px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Strategy Registry ────────────────────────────────────────────────────────

STOCK_STRATEGIES = {
    "MA Crossover (均线交叉)": {
        "class": MACrossoverStrategy,
        "params": {
            "fast_window": {"default": 10, "min": 3, "max": 50, "step": 1},
            "slow_window": {"default": 30, "min": 10, "max": 100, "step": 1},
            "use_ema": {"default": False},
        },
    },
    "Momentum / RSI (动量)": {
        "class": MomentumStrategy,
        "params": {
            "rsi_period": {"default": 14, "min": 5, "max": 30, "step": 1},
            "oversold": {"default": 30, "min": 10, "max": 45, "step": 1},
            "overbought": {"default": 70, "min": 55, "max": 90, "step": 1},
        },
    },
    "Mean Reversion (均值回归)": {
        "class": MeanReversionStrategy,
        "params": {
            "window": {"default": 20, "min": 5, "max": 50, "step": 1},
            "num_std": {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1},
        },
    },
    "Dual Thrust (双推力)": {
        "class": DualThrustStrategy,
        "params": {
            "lookback": {"default": 4, "min": 2, "max": 10, "step": 1},
            "k_up": {"default": 0.7, "min": 0.3, "max": 1.2, "step": 0.05},
            "k_down": {"default": 0.7, "min": 0.3, "max": 1.2, "step": 0.05},
        },
    },
}


# ── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar():
    """Render the sidebar and return user selections."""
    st.sidebar.markdown("## ⚙️ Configuration")

    mode = st.sidebar.radio(
        "Mode / 模式",
        ["Stock Strategies", "Option Strategies", "Live Simulation", "Alpaca Live"],
        index=0,
    )

    st.sidebar.markdown("---")

    # Data settings
    st.sidebar.markdown("### Data Settings")
    data_source = st.sidebar.selectbox(
        "Data Source / 数据源",
        ["Synthetic (GBM)", "Regime Switching", "Alpaca API"],
    )

    symbol = st.sidebar.selectbox("Symbol / 标的", ["SPY", "QQQ"])
    days = st.sidebar.slider("Trading Days / 交易天数", 200, 2000, 750, step=50)
    seed = st.sidebar.number_input("Random Seed / 随机种子", 0, 9999, 42)
    capital = st.sidebar.number_input(
        "Initial Capital ($) / 初始资金",
        10_000, 10_000_000, 100_000, step=10_000,
    )
    commission = st.sidebar.slider(
        "Commission Rate / 手续费", 0.0, 0.01, 0.001, step=0.0001, format="%.4f"
    )

    return {
        "mode": mode,
        "data_source": data_source,
        "symbol": symbol,
        "days": days,
        "seed": seed,
        "capital": capital,
        "commission": commission,
    }


# ── Data Loading ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_data(source: str, symbol: str, days: int, seed: int) -> MarketData:
    """Load market data from selected source."""
    if source == "Alpaca API":
        try:
            from data.alpaca_client import AlpacaClient, AlpacaConfig
            config = AlpacaConfig.from_env()
            if config.is_configured:
                client = AlpacaClient(config)
                return client.fetch_market_data(symbol, days=days)
        except Exception as e:
            st.warning(f"Alpaca fetch failed: {e}. Falling back to synthetic data.")
    if source == "Regime Switching":
        return generate_regime_data(days=days, seed=seed)
    # Default: synthetic
    if symbol == "QQQ":
        return generate_synthetic_data(
            symbol="QQQ", days=days, start_price=420.0,
            volatility=0.020, trend=0.0005, seed=seed,
        )
    return generate_synthetic_data(
        symbol="SPY", days=days, start_price=480.0,
        volatility=0.015, trend=0.0004, seed=seed,
    )


# ── Chart Builders ───────────────────────────────────────────────────────────

CHART_COLORS = {
    "equity": "#00D4AA",
    "benchmark": "#5C6BC0",
    "drawdown": "#FF5252",
    "buy": "#00E676",
    "sell": "#FF5252",
    "grid": "#1E2433",
    "bg": "#0E1117",
    "paper": "#0E1117",
}


def build_equity_chart(result: BacktestResult, data: MarketData) -> go.Figure:
    """Build Plotly equity curve with drawdown and buy/sell markers."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
        vertical_spacing=0.04,
        subplot_titles=["Equity Curve / 资金曲线", "Drawdown / 回撤"],
    )

    dates = result.dates

    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=dates, y=result.portfolio_values,
            name="Portfolio", line=dict(color=CHART_COLORS["equity"], width=2),
            fill="tozeroy", fillcolor="rgba(0,212,170,0.08)",
        ),
        row=1, col=1,
    )

    # Benchmark (buy & hold)
    if data.close:
        scale = result.initial_capital / data.close[0]
        benchmark = [round(c * scale, 2) for c in data.close]
        bench_dates = data.dates[:len(benchmark)]
        fig.add_trace(
            go.Scatter(
                x=bench_dates, y=benchmark,
                name="Buy & Hold", line=dict(color=CHART_COLORS["benchmark"],
                                             width=1, dash="dot"),
            ),
            row=1, col=1,
        )

    # Buy / Sell markers
    buy_dates = [dates[i] for i in range(len(dates)) if result.signals[i] == 1]
    buy_vals = [result.portfolio_values[i] for i in range(len(dates))
                if result.signals[i] == 1]
    sell_dates = [dates[i] for i in range(len(dates)) if result.signals[i] == -1]
    sell_vals = [result.portfolio_values[i] for i in range(len(dates))
                 if result.signals[i] == -1]

    if buy_dates:
        fig.add_trace(
            go.Scatter(
                x=buy_dates, y=buy_vals, mode="markers", name="Buy",
                marker=dict(color=CHART_COLORS["buy"], size=7, symbol="triangle-up"),
            ),
            row=1, col=1,
        )
    if sell_dates:
        fig.add_trace(
            go.Scatter(
                x=sell_dates, y=sell_vals, mode="markers", name="Sell",
                marker=dict(color=CHART_COLORS["sell"], size=7, symbol="triangle-down"),
            ),
            row=1, col=1,
        )

    # Drawdown
    pv = result.portfolio_values
    peak = pv[0]
    dd = []
    for v in pv:
        if v > peak:
            peak = v
        dd.append((v - peak) / peak * 100 if peak > 0 else 0)

    fig.add_trace(
        go.Scatter(
            x=dates, y=dd, name="Drawdown %",
            line=dict(color=CHART_COLORS["drawdown"], width=1),
            fill="tozeroy", fillcolor="rgba(255,82,82,0.15)",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        height=520,
        template="plotly_dark",
        paper_bgcolor=CHART_COLORS["paper"],
        plot_bgcolor=CHART_COLORS["bg"],
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        margin=dict(l=50, r=20, t=50, b=30),
        xaxis2=dict(gridcolor=CHART_COLORS["grid"]),
        yaxis=dict(gridcolor=CHART_COLORS["grid"]),
        yaxis2=dict(gridcolor=CHART_COLORS["grid"], ticksuffix="%"),
    )

    return fig


def build_price_chart(data: MarketData) -> go.Figure:
    """Build OHLC candlestick chart."""
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=data.dates,
        open=data.open, high=data.high, low=data.low, close=data.close,
        increasing_line_color="#00D4AA", decreasing_line_color="#FF5252",
        name=f"Price",
    ))

    fig.update_layout(
        height=400,
        template="plotly_dark",
        paper_bgcolor=CHART_COLORS["paper"],
        plot_bgcolor=CHART_COLORS["bg"],
        margin=dict(l=50, r=20, t=30, b=30),
        xaxis=dict(gridcolor=CHART_COLORS["grid"], rangeslider=dict(visible=False)),
        yaxis=dict(gridcolor=CHART_COLORS["grid"], tickprefix="$"),
    )

    return fig


def build_option_equity_chart(result) -> go.Figure:
    """Equity curve for option backtest results."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
        vertical_spacing=0.04,
        subplot_titles=["Option Portfolio / 期权组合净值", "Drawdown / 回撤"],
    )

    dates = result.dates

    fig.add_trace(
        go.Scatter(
            x=dates, y=result.portfolio_values,
            name="Portfolio", line=dict(color="#00D4AA", width=2),
            fill="tozeroy", fillcolor="rgba(0,212,170,0.08)",
        ),
        row=1, col=1,
    )

    # Drawdown
    peak = result.portfolio_values[0]
    dd = []
    for v in result.portfolio_values:
        if v > peak:
            peak = v
        dd.append((v - peak) / peak * 100 if peak > 0 else 0)

    fig.add_trace(
        go.Scatter(
            x=dates, y=dd, name="Drawdown %",
            line=dict(color="#FF5252", width=1),
            fill="tozeroy", fillcolor="rgba(255,82,82,0.15)",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        height=520,
        template="plotly_dark",
        paper_bgcolor=CHART_COLORS["paper"],
        plot_bgcolor=CHART_COLORS["bg"],
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        margin=dict(l=50, r=20, t=50, b=30),
        yaxis=dict(gridcolor=CHART_COLORS["grid"], tickprefix="$"),
        yaxis2=dict(gridcolor=CHART_COLORS["grid"], ticksuffix="%"),
    )

    return fig


# ── Page: Stock Strategies ───────────────────────────────────────────────────

def page_stock(cfg):
    st.markdown("# 📊 Stock Strategy Backtester")

    data = load_data(cfg["data_source"], cfg["symbol"], cfg["days"], cfg["seed"])
    st.caption(
        f"Data: {data.dates[0]} → {data.dates[-1]} | "
        f"{len(data)} bars | "
        f"Price: ${min(data.close):.2f} – ${max(data.close):.2f}"
    )

    # Strategy selection + params
    st.markdown("---")
    col_strat, col_params = st.columns([1, 2])

    with col_strat:
        strat_name = st.selectbox("Strategy / 策略", list(STOCK_STRATEGIES.keys()))
        compare_all = st.checkbox("Compare All / 对比全部", value=False)

    strat_info = STOCK_STRATEGIES[strat_name]
    param_values = {}
    with col_params:
        st.markdown("**Parameters / 参数**")
        param_cols = st.columns(len(strat_info["params"]))
        for i, (pname, pconf) in enumerate(strat_info["params"].items()):
            with param_cols[i]:
                if isinstance(pconf["default"], bool):
                    param_values[pname] = st.checkbox(pname, value=pconf["default"])
                elif isinstance(pconf["default"], float):
                    param_values[pname] = st.slider(
                        pname, float(pconf["min"]), float(pconf["max"]),
                        float(pconf["default"]), step=float(pconf["step"]),
                    )
                else:
                    param_values[pname] = st.slider(
                        pname, int(pconf["min"]), int(pconf["max"]),
                        int(pconf["default"]), step=int(pconf["step"]),
                    )

    if st.button("🚀 Run Backtest / 运行回测", type="primary", use_container_width=True):
        engine = BacktestEngine(
            initial_capital=cfg["capital"],
            commission_rate=cfg["commission"],
        )

        if compare_all:
            _run_stock_compare(engine, data, cfg)
        else:
            _run_single_stock(engine, data, strat_info, param_values, strat_name, cfg)


def _run_single_stock(engine, data, strat_info, params, name, cfg):
    """Run a single stock strategy and display results."""
    strategy = strat_info["class"](**params)
    with st.spinner("Running backtest..."):
        result = engine.run(strategy, data)
    metrics = calculate_all_metrics(result)
    ratings = rate_strategy(metrics)

    # KPI Row
    st.markdown("---")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Return", metrics.get("total_return_pct", "N/A"))
    k2.metric("Annual Return", metrics.get("annualized_return_pct", "N/A"))
    k3.metric("Sharpe Ratio", metrics.get("sharpe_ratio", 0))
    k4.metric("Max Drawdown", metrics.get("max_drawdown_pct", "N/A"))
    k5.metric("Win Rate", metrics.get("win_rate_pct", "N/A"))
    k6.metric("Profit Factor", metrics.get("profit_factor", "N/A"))

    # Equity chart
    st.plotly_chart(build_equity_chart(result, data), use_container_width=True)

    # Details tabs
    tab_trades, tab_price, tab_rating = st.tabs(
        ["Trade Log / 交易记录", "Price Chart / K线图", "Rating / 评级"]
    )

    with tab_trades:
        if result.trades:
            rows = []
            for t in result.trades:
                rows.append({
                    "Entry": t.entry_date,
                    "Exit": t.exit_date,
                    "Entry $": f"${t.entry_price:.2f}",
                    "Exit $": f"${t.exit_price:.2f}",
                    "Shares": t.shares,
                    "P&L": f"${t.pnl:+,.2f}",
                    "P&L %": f"{t.pnl_pct * 100:+.1f}%",
                    "Days": t.holding_days,
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No trades generated.")

    with tab_price:
        st.plotly_chart(build_price_chart(data), use_container_width=True)

    with tab_rating:
        for label, grade in ratings.items():
            st.markdown(f"**{label}**: {grade}")


def _run_stock_compare(engine, data, cfg):
    """Run all stock strategies and show comparison."""
    all_results = []
    progress = st.progress(0)
    total = len(STOCK_STRATEGIES)

    for idx, (name, info) in enumerate(STOCK_STRATEGIES.items()):
        defaults = {k: v["default"] for k, v in info["params"].items()}
        strategy = info["class"](**defaults)
        result = engine.run(strategy, data)
        all_results.append((name, result))
        progress.progress((idx + 1) / total)

    progress.empty()

    # Comparison table
    comp_rows = []
    for name, result in all_results:
        m = calculate_all_metrics(result)
        comp_rows.append({
            "Strategy": name,
            "Return": m.get("total_return_pct", "N/A"),
            "Annual": m.get("annualized_return_pct", "N/A"),
            "Sharpe": m.get("sharpe_ratio", 0),
            "Sortino": m.get("sortino_ratio", 0),
            "Max DD": m.get("max_drawdown_pct", "N/A"),
            "Win Rate": m.get("win_rate_pct", "N/A"),
            "Trades": m.get("total_trades", 0),
            "Profit Factor": m.get("profit_factor", "N/A"),
        })
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    # Overlay equity curves
    fig = go.Figure()
    colors = ["#00D4AA", "#5C6BC0", "#FFB74D", "#E040FB"]
    for idx, (name, result) in enumerate(all_results):
        fig.add_trace(go.Scatter(
            x=result.dates, y=result.portfolio_values,
            name=name, line=dict(color=colors[idx % len(colors)], width=2),
        ))
    fig.update_layout(
        height=450, template="plotly_dark",
        paper_bgcolor=CHART_COLORS["paper"],
        plot_bgcolor=CHART_COLORS["bg"],
        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
        margin=dict(l=50, r=20, t=30, b=30),
        yaxis=dict(gridcolor=CHART_COLORS["grid"], tickprefix="$"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Page: Option Strategies ──────────────────────────────────────────────────

def page_options(cfg):
    st.markdown("# 🎯 Option Strategy Backtester")

    try:
        from options.backtest import OptionBacktestEngine
        from options.option_data import generate_spy_data, generate_qqq_data
        from options.strategies import (
            IronCondorStrategy, VerticalSpreadStrategy, WheelStrategy,
            StraddleStrategy, IVAdaptiveStrategy,
            ButterflySpreadStrategy, PCREnhancedStrategy, TADrivenStrategy,
        )
    except ImportError as e:
        st.error(f"Options module import error: {e}")
        return

    OPTION_STRATS = {
        "Iron Condor (铁鹰)": {
            "class": IronCondorStrategy,
            "defaults": {"iv_rank_entry": 40.0, "dte_target": 45,
                         "short_delta": 0.16, "wing_width": 5.0},
        },
        "Vertical Spread (垂直价差)": {
            "class": VerticalSpreadStrategy,
            "defaults": {"ma_fast": 20, "ma_slow": 50,
                         "iv_rank_entry": 30.0, "spread_width": 5.0},
        },
        "Wheel (轮动)": {
            "class": WheelStrategy,
            "defaults": {"iv_rank_entry": 30.0, "dte_target": 30,
                         "put_delta": 0.30, "call_delta": 0.30},
        },
        "Long Strangle (宽跨式)": {
            "class": StraddleStrategy,
            "defaults": {"iv_rank_entry": 30.0, "dte_target": 45,
                         "use_strangle": True, "strangle_width": 5.0},
        },
        "IV-Adaptive (IV自适应)": {
            "class": IVAdaptiveStrategy,
            "defaults": {"high_iv_threshold": 60.0, "low_iv_threshold": 25.0,
                         "dte_target": 30},
        },
        "Butterfly (蝶式)": {
            "class": ButterflySpreadStrategy,
            "defaults": {"low_iv_threshold": 10.0, "high_iv_threshold": 70.0,
                         "dte_target": 35, "wing_width": 5.0,
                         "profit_target": 0.75},
        },
        "PCR-Enhanced IC (PCR增强铁鹰)": {
            "class": PCREnhancedStrategy,
            "defaults": {"iv_rank_entry": 40.0, "dte_target": 45,
                         "short_delta": 0.16, "wing_width": 5.0,
                         "require_pcr": True, "require_vix_band": True},
        },
        "TA-Driven (技术分析)": {
            "class": TADrivenStrategy,
            "defaults": {"dte_target": 30, "profit_target": 0.50,
                         "stop_loss": 0.50},
        },
    }

    # Generate data
    if cfg["symbol"] == "QQQ":
        data = generate_qqq_data(days=cfg["days"], seed=cfg["seed"])
    else:
        data = generate_spy_data(days=cfg["days"], seed=cfg["seed"])

    st.caption(
        f"Underlying: {cfg['symbol']} | {data.dates[0]} → {data.dates[-1]} | "
        f"{len(data)} bars"
    )

    # Strategy selection
    col_s, col_a = st.columns([2, 1])
    with col_s:
        opt_name = st.selectbox("Option Strategy / 期权策略", list(OPTION_STRATS.keys()))
    with col_a:
        compare_opts = st.checkbox("Compare All Options / 对比全部期权策略")

    engine = OptionBacktestEngine(
        initial_capital=cfg["capital"],
        commission_per_contract=0.65,
    )

    if st.button("🚀 Run Option Backtest / 运行期权回测", type="primary",
                  use_container_width=True):
        if compare_opts:
            _run_option_compare(engine, data, OPTION_STRATS)
        else:
            _run_single_option(engine, data, OPTION_STRATS[opt_name], opt_name)

    # Option chain viewer
    st.markdown("---")
    with st.expander("📋 Option Chain Explorer / 期权链查看器"):
        _show_option_chain(data, cfg)


def _run_single_option(engine, data, info, name):
    """Run single option strategy."""
    strategy = info["class"](**info["defaults"])
    with st.spinner("Running option backtest..."):
        result = engine.run(strategy, data)

    # KPIs
    st.markdown("---")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Return", f"{result.total_return_pct:+.2f}%")
    k2.metric("Sharpe", f"{result.sharpe_ratio:.2f}")
    k3.metric("Max DD", f"{result.max_drawdown_pct:.2f}%")
    k4.metric("Win Rate", f"{result.win_rate:.1f}%")
    k5.metric("Trades", f"{result.positions_closed}")
    k6.metric("Calmar", f"{result.calmar_ratio:.2f}")

    st.plotly_chart(build_option_equity_chart(result), use_container_width=True)

    # Trade log
    tab_log, tab_cost, tab_ev = st.tabs(
        ["Trade Log / 交易记录", "Costs / 成本分析", "EV Analysis / 期望值"]
    )

    with tab_log:
        if result.trades:
            rows = []
            for t in result.trades:
                rows.append({
                    "Entry": t.entry_date,
                    "Exit": t.exit_date,
                    "Type": t.strategy_name,
                    "Legs": t.legs_description,
                    "P&L": f"${t.exit_pnl:+,.0f}",
                    "P&L %": f"{t.pnl_pct:+.1f}%",
                    "POP": f"{t.pop_at_entry:.0%}",
                    "EV": f"${t.ev_at_entry:+,.0f}",
                    "Days": t.holding_days,
                    "Reason": t.exit_reason,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No completed option trades.")

    with tab_cost:
        c1, c2, c3 = st.columns(3)
        c1.metric("Commission", f"${result.total_commission:,.2f}")
        c2.metric("Slippage", f"${result.total_slippage:,.2f}")
        c3.metric("Total Costs", f"${result.total_transaction_costs:,.2f}")
        cost_pct = (result.total_transaction_costs / result.initial_capital * 100
                    if result.initial_capital > 0 else 0)
        st.progress(min(cost_pct / 5, 1.0),
                    text=f"Cost drag: {cost_pct:.2f}% of capital")

    with tab_ev:
        st.metric("Avg EV at Entry", f"${result.avg_ev_at_entry:,.2f}")
        st.metric("Avg POP at Entry", f"{result.avg_pop_at_entry:.1%}")
        st.metric("EV-Filtered (rejected)", f"{result.ev_filtered_trades}")


def _run_option_compare(engine, data, strats):
    """Compare all option strategies."""
    all_results = []
    progress = st.progress(0)
    total = len(strats)

    for idx, (name, info) in enumerate(strats.items()):
        strategy = info["class"](**info["defaults"])
        result = engine.run(strategy, data)
        all_results.append((name, result))
        progress.progress((idx + 1) / total)

    progress.empty()

    # Comparison table
    comp_rows = []
    for name, r in all_results:
        comp_rows.append({
            "Strategy": name,
            "Return": f"{r.total_return_pct:+.2f}%",
            "Sharpe": f"{r.sharpe_ratio:.2f}",
            "Sortino": f"{r.sortino_ratio:.2f}",
            "Calmar": f"{r.calmar_ratio:.2f}",
            "Max DD": f"{r.max_drawdown_pct:.2f}%",
            "Win Rate": f"{r.win_rate:.1f}%",
            "Trades": r.positions_closed,
            "Profit Factor": f"{r.profit_factor:.2f}",
            "Avg EV": f"${r.avg_ev_at_entry:+,.0f}",
            "Costs": f"${r.total_transaction_costs:,.0f}",
        })

    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    # Overlay equity curves
    fig = go.Figure()
    colors = ["#00D4AA", "#5C6BC0", "#FFB74D", "#E040FB",
              "#26A69A", "#AB47BC", "#FFA726"]
    for idx, (name, r) in enumerate(all_results):
        fig.add_trace(go.Scatter(
            x=r.dates, y=r.portfolio_values,
            name=name, line=dict(color=colors[idx % len(colors)], width=2),
        ))
    fig.update_layout(
        height=450, template="plotly_dark",
        paper_bgcolor=CHART_COLORS["paper"],
        plot_bgcolor=CHART_COLORS["bg"],
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        margin=dict(l=50, r=20, t=30, b=30),
        yaxis=dict(gridcolor=CHART_COLORS["grid"], tickprefix="$"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _show_option_chain(data, cfg):
    """Render an interactive option chain viewer."""
    from options.option_data import generate_option_chain

    price = data.close[-1]
    date = str(data.dates[-1])
    dte = st.slider("DTE / 到期天数", 7, 90, 30)

    from datetime import datetime, timedelta
    exp_date = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=dte)).strftime("%Y-%m-%d")

    chain = generate_option_chain(
        symbol=cfg["symbol"], underlying_price=price,
        current_date=date, expiration_date=exp_date,
        base_iv=0.20, num_strikes=10, seed=42,
    )

    rows = []
    for strike in chain.strikes:
        c = chain.get_call(strike)
        p = chain.get_put(strike)
        if c and p:
            rows.append({
                "C.Bid": f"{c.bid:.2f}",
                "C.Ask": f"{c.ask:.2f}",
                "C.Delta": f"{c.delta:+.3f}",
                "C.IV": f"{c.iv:.1%}",
                "Strike": f"{strike:.0f}",
                "P.Bid": f"{p.bid:.2f}",
                "P.Ask": f"{p.ask:.2f}",
                "P.Delta": f"{p.delta:+.3f}",
                "P.IV": f"{p.iv:.1%}",
            })

    df = pd.DataFrame(rows)
    st.markdown(f"**{cfg['symbol']} @ ${price:.2f} | Exp: {exp_date} ({dte} DTE)**")
    st.dataframe(df, use_container_width=True, hide_index=True)


# ── Page: Alpaca Live ────────────────────────────────────────────────────────

def page_alpaca(cfg):
    st.markdown("# 🏦 Alpaca API Integration")

    try:
        from data.alpaca_client import (
            AlpacaClient, AlpacaConfig, AlpacaAPIError,
            test_connection, create_client,
        )
    except ImportError as e:
        st.error(f"Failed to import alpaca_client: {e}")
        return

    config = AlpacaConfig.from_env()

    # Connection status
    st.markdown("### Connection Status / 连接状态")
    col_key, col_url = st.columns(2)
    with col_key:
        api_key_display = (config.api_key[:8] + "..." if config.api_key else "Not Set")
        st.text_input("API Key", value=api_key_display, disabled=True)
    with col_url:
        st.text_input("Base URL", value=config.base_url, disabled=True)

    if not config.is_configured:
        st.warning(
            "Alpaca API not configured. Set environment variables:\n\n"
            "```bash\n"
            "export ALPACA_API_KEY='your-key'\n"
            "export ALPACA_SECRET_KEY='your-secret'\n"
            "export ALPACA_BASE_URL='https://paper-api.alpaca.markets'\n"
            "```"
        )
        return

    if st.button("🔌 Test Connection / 测试连接", type="primary"):
        with st.spinner("Connecting to Alpaca..."):
            ok, msg = test_connection()
        if ok:
            st.success(msg)
        else:
            st.error(msg)

    st.markdown("---")

    # Tabs for different Alpaca features
    tab_acct, tab_data, tab_order, tab_pos = st.tabs(
        ["Account / 账户", "Market Data / 行情", "Orders / 订单", "Positions / 持仓"]
    )

    client = AlpacaClient(config)

    with tab_acct:
        if st.button("Refresh Account / 刷新账户"):
            try:
                acct = client.get_account()
                a1, a2, a3, a4 = st.columns(4)
                a1.metric("Equity", f"${float(acct.get('equity', 0)):,.2f}")
                a2.metric("Buying Power", f"${float(acct.get('buying_power', 0)):,.2f}")
                a3.metric("Cash", f"${float(acct.get('cash', 0)):,.2f}")
                a4.metric("Status", acct.get("status", "N/A"))

                mode = "PAPER" if config.is_paper else "LIVE"
                st.info(f"Trading Mode: **{mode}**")
            except Exception as e:
                st.error(f"Error: {e}")

    with tab_data:
        col_sym, col_tf, col_lim = st.columns(3)
        with col_sym:
            sym = st.text_input("Symbol", value="SPY")
        with col_tf:
            tf = st.selectbox("Timeframe", ["1Day", "1Hour", "1Min"])
        with col_lim:
            limit = st.number_input("Bars", 50, 1000, 200)

        if st.button("Fetch Bars / 获取K线"):
            try:
                with st.spinner(f"Fetching {sym} bars..."):
                    bars = client.get_bars(sym, timeframe=tf, limit=limit)

                if bars:
                    rows = []
                    for b in bars:
                        rows.append({
                            "Date": b.get("t", "")[:10],
                            "Open": float(b.get("o", 0)),
                            "High": float(b.get("h", 0)),
                            "Low": float(b.get("l", 0)),
                            "Close": float(b.get("c", 0)),
                            "Volume": int(b.get("v", 0)),
                        })
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    # Quick chart
                    fig = go.Figure(go.Candlestick(
                        x=df["Date"], open=df["Open"],
                        high=df["High"], low=df["Low"], close=df["Close"],
                        increasing_line_color="#00D4AA",
                        decreasing_line_color="#FF5252",
                    ))
                    fig.update_layout(
                        height=400, template="plotly_dark",
                        paper_bgcolor=CHART_COLORS["paper"],
                        plot_bgcolor=CHART_COLORS["bg"],
                        xaxis=dict(rangeslider=dict(visible=False)),
                        margin=dict(l=50, r=20, t=20, b=30),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No bars returned.")
            except Exception as e:
                st.error(f"Error fetching bars: {e}")

    with tab_order:
        st.markdown("**Paper Trading Orders / 模拟下单**")
        if not config.is_paper:
            st.error("Live trading detected! Switch to paper mode for safety.")
        else:
            o1, o2, o3, o4 = st.columns(4)
            with o1:
                order_sym = st.text_input("Order Symbol", value="SPY", key="order_sym")
            with o2:
                order_qty = st.number_input("Qty", 1, 1000, 1, key="order_qty")
            with o3:
                order_side = st.selectbox("Side", ["buy", "sell"], key="order_side")
            with o4:
                order_type = st.selectbox(
                    "Type", ["market", "limit"], key="order_type"
                )

            limit_price = None
            if order_type == "limit":
                limit_price = st.number_input("Limit Price", 1.0, 99999.0, 450.0)

            if st.button("Submit Order / 提交订单", type="primary"):
                try:
                    resp = client.submit_order(
                        order_sym, order_qty, order_side,
                        order_type=order_type, limit_price=limit_price,
                    )
                    st.success(f"Order submitted! ID: {resp.get('id', 'N/A')}")
                    st.json(resp)
                except Exception as e:
                    st.error(f"Order failed: {e}")

            st.markdown("---")
            if st.button("View Open Orders / 查看挂单"):
                try:
                    orders = client.get_orders(status="open")
                    if orders:
                        st.dataframe(
                            pd.DataFrame(orders)[
                                ["id", "symbol", "side", "qty", "type",
                                 "status", "created_at"]
                            ] if len(orders) > 0 else pd.DataFrame(),
                            use_container_width=True, hide_index=True,
                        )
                    else:
                        st.info("No open orders.")
                except Exception as e:
                    st.error(f"Error: {e}")

    with tab_pos:
        if st.button("View Positions / 查看持仓"):
            try:
                positions = client.get_positions()
                if positions:
                    rows = []
                    for p in positions:
                        rows.append({
                            "Symbol": p.get("symbol"),
                            "Qty": p.get("qty"),
                            "Avg Entry": f"${float(p.get('avg_entry_price', 0)):,.2f}",
                            "Current": f"${float(p.get('current_price', 0)):,.2f}",
                            "P&L": f"${float(p.get('unrealized_pl', 0)):+,.2f}",
                            "P&L %": f"{float(p.get('unrealized_plpc', 0)) * 100:+.2f}%",
                        })
                    st.dataframe(
                        pd.DataFrame(rows), use_container_width=True, hide_index=True
                    )
                else:
                    st.info("No open positions.")
            except Exception as e:
                st.error(f"Error: {e}")


# ── Page: Live Simulation ────────────────────────────────────────────────────

def page_live_sim(cfg):
    st.markdown("# 🔴 Live Market Simulation")
    st.caption(
        "Simulates real-time trading using recent synthetic data "
        "(or Alpaca API if configured). Runs strategies on current market conditions."
    )

    try:
        from data.live_simulator import LiveSimulator, generate_live_simulation
    except ImportError as e:
        st.error(f"Failed to import live_simulator: {e}")
        return

    try:
        from options.strategies import (
            IronCondorStrategy, VerticalSpreadStrategy, WheelStrategy,
            StraddleStrategy, IVAdaptiveStrategy,
            ButterflySpreadStrategy, PCREnhancedStrategy, TADrivenStrategy,
        )
    except ImportError as e:
        st.error(f"Options strategies import error: {e}")
        return

    LIVE_STRATS = {
        "Iron Condor": IronCondorStrategy,
        "Vertical Spread": VerticalSpreadStrategy,
        "Wheel": WheelStrategy,
        "Long Strangle": StraddleStrategy,
        "IV-Adaptive": IVAdaptiveStrategy,
        "Butterfly": ButterflySpreadStrategy,
        "PCR-Enhanced IC": PCREnhancedStrategy,
        "TA-Driven": TADrivenStrategy,
    }

    col_s, col_d = st.columns(2)
    with col_s:
        strat_name = st.selectbox("Strategy / 策略", list(LIVE_STRATS.keys()))
    with col_d:
        sim_days = st.slider("Simulation Days / 模拟天数", 200, 1500, 750)

    if st.button("🔴 Run Live Simulation / 运行实时模拟", type="primary",
                  use_container_width=True):
        with st.spinner("Running live simulation..."):
            sim = LiveSimulator(
                symbol=cfg["symbol"],
                initial_capital=cfg["capital"],
            )

            data = sim.fetch_current_data(days=sim_days, seed=cfg["seed"])
            strategy = LIVE_STRATS[strat_name]()
            result = sim.run_live_backtest(strategy, data)

        # Market snapshot
        st.markdown("---")
        st.markdown("### Market Snapshot / 市场快照")
        snapshots = sim.get_snapshots(data)
        if snapshots:
            latest = snapshots[-1]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Price", f"${latest.price:.2f}",
                       delta=f"{latest.change_pct:+.2f}%")
            m2.metric("IV Estimate", f"{latest.iv_estimate:.1%}")
            m3.metric("IV Rank", f"{latest.ivr_estimate:.0f}")
            m4.metric("Date", latest.timestamp[:10])

        # KPIs
        st.markdown("### Strategy Performance / 策略表现")
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Return", f"{result.total_return_pct:+.2f}%")
        k2.metric("Sharpe", f"{result.sharpe_ratio:.2f}")
        k3.metric("Max DD", f"{result.max_drawdown_pct:.2f}%")
        k4.metric("Win Rate", f"{result.win_rate:.1f}%")
        k5.metric("Trades", f"{result.positions_closed}")
        k6.metric("Calmar", f"{result.calmar_ratio:.2f}")

        # Equity chart
        st.plotly_chart(build_option_equity_chart(result), use_container_width=True)

        # Price chart with snapshots
        if snapshots:
            fig_price = go.Figure()
            snap_dates = [s.timestamp[:10] for s in snapshots]
            snap_prices = [s.price for s in snapshots]
            snap_iv = [s.iv_estimate * 100 for s in snapshots]

            fig_price.add_trace(go.Scatter(
                x=snap_dates, y=snap_prices, name="Price",
                line=dict(color="#00D4AA", width=2),
            ))
            fig_price.add_trace(go.Scatter(
                x=snap_dates, y=snap_iv, name="IV %",
                yaxis="y2", line=dict(color="#FFB74D", width=1, dash="dot"),
            ))
            fig_price.update_layout(
                height=350, template="plotly_dark",
                paper_bgcolor=CHART_COLORS["paper"],
                plot_bgcolor=CHART_COLORS["bg"],
                margin=dict(l=50, r=50, t=20, b=30),
                yaxis=dict(title="Price", gridcolor=CHART_COLORS["grid"],
                           tickprefix="$"),
                yaxis2=dict(title="IV %", overlaying="y", side="right",
                            gridcolor=CHART_COLORS["grid"], ticksuffix="%"),
                legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
            )
            st.plotly_chart(fig_price, use_container_width=True)

        # Trade log
        if result.trades:
            st.markdown("### Trade Log / 交易记录")
            rows = []
            for t in result.trades:
                rows.append({
                    "Entry": t.entry_date,
                    "Exit": t.exit_date,
                    "Type": t.strategy_name,
                    "P&L": f"${t.exit_pnl:+,.0f}",
                    "P&L %": f"{t.pnl_pct:+.1f}%",
                    "Days": t.holding_days,
                    "Reason": t.exit_reason,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                          hide_index=True)


# ── Main App ─────────────────────────────────────────────────────────────────

def main():
    cfg = render_sidebar()

    if cfg["mode"] == "Stock Strategies":
        page_stock(cfg)
    elif cfg["mode"] == "Option Strategies":
        page_options(cfg)
    elif cfg["mode"] == "Live Simulation":
        page_live_sim(cfg)
    elif cfg["mode"] == "Alpaca Live":
        page_alpaca(cfg)


if __name__ == "__main__":
    main()
