"""
报告生成 - Report Generator

生成文本报告和ASCII图表来检测策略表现。
Generates text reports and ASCII charts for strategy evaluation.
"""

from typing import List, Dict, Any
from engine.backtest import BacktestResult
from evaluation.metrics import calculate_all_metrics, rate_strategy


def generate_report(result: BacktestResult) -> str:
    """
    生成完整的策略评估报告 / Generate full strategy evaluation report.
    """
    metrics = calculate_all_metrics(result)
    ratings = rate_strategy(metrics)

    lines = []
    w = 60  # report width

    lines.append("=" * w)
    lines.append(f"  策略回测报告 / Backtest Report")
    lines.append(f"  Strategy: {result.strategy_name}")
    lines.append("=" * w)

    # Parameters
    lines.append("\n【策略参数 / Parameters】")
    for k, v in result.params.items():
        lines.append(f"  {k}: {v}")

    # Return metrics
    lines.append("\n【收益指标 / Return Metrics】")
    lines.append(f"  初始资金 Initial Capital:    ${result.initial_capital:>12,.2f}")
    lines.append(f"  最终价值 Final Value:        ${result.final_value:>12,.2f}")
    lines.append(f"  总收益率 Total Return:       {metrics.get('total_return_pct', 'N/A'):>12s}")
    lines.append(f"  年化收益 Annualized Return:  {metrics.get('annualized_return_pct', 'N/A'):>12s}")

    # Risk metrics
    lines.append("\n【风险指标 / Risk Metrics】")
    lines.append(f"  年化波动 Annual Volatility:  {metrics.get('annual_volatility_pct', 'N/A'):>12s}")
    lines.append(f"  最大回撤 Max Drawdown:       {metrics.get('max_drawdown_pct', 'N/A'):>12s}")
    dd_period = metrics.get('max_drawdown_period', 'N/A')
    lines.append(f"  回撤区间 DD Period:          {dd_period}")

    # Risk-adjusted
    lines.append("\n【风险调整指标 / Risk-Adjusted Metrics】")
    lines.append(f"  夏普比率 Sharpe Ratio:       {metrics.get('sharpe_ratio', 'N/A'):>12}")
    lines.append(f"  索提诺比 Sortino Ratio:      {metrics.get('sortino_ratio', 'N/A'):>12}")
    lines.append(f"  卡玛比率 Calmar Ratio:       {metrics.get('calmar_ratio', 'N/A'):>12}")

    # Trade stats
    lines.append("\n【交易统计 / Trade Statistics】")
    lines.append(f"  总交易数 Total Trades:       {metrics.get('total_trades', 0):>12}")
    lines.append(f"  盈利交易 Winning:            {metrics.get('winning_trades', 0):>12}")
    lines.append(f"  亏损交易 Losing:             {metrics.get('losing_trades', 0):>12}")
    lines.append(f"  胜率     Win Rate:           {metrics.get('win_rate_pct', 'N/A'):>12s}")
    lines.append(f"  盈亏比   Profit Factor:      {metrics.get('profit_factor', 'N/A'):>12}")
    lines.append(f"  均盈利   Avg Win:            ${metrics.get('avg_win', 0):>11,.2f}")
    lines.append(f"  均亏损   Avg Loss:           ${metrics.get('avg_loss', 0):>11,.2f}")
    lines.append(f"  持仓天数 Avg Holding Days:   {metrics.get('avg_holding_days', 'N/A'):>12}")
    lines.append(f"  最佳交易 Best Trade:         {metrics.get('best_trade', 'N/A')}")
    lines.append(f"  最差交易 Worst Trade:        {metrics.get('worst_trade', 'N/A')}")
    lines.append(f"  最大连胜 Max Consec Wins:    {metrics.get('max_consecutive_wins', 0):>12}")
    lines.append(f"  最大连亏 Max Consec Losses:  {metrics.get('max_consecutive_losses', 0):>12}")
    lines.append(f"  总手续费 Total Commission:   ${metrics.get('total_commission', 0):>11,.2f}")

    # Ratings
    lines.append("\n【策略评级 / Strategy Rating】")
    lines.append(f"  夏普比率: {ratings.get('sharpe', 'N/A')}")
    lines.append(f"  回撤风险: {ratings.get('drawdown', 'N/A')}")
    lines.append(f"  交易质量: {ratings.get('trade_quality', 'N/A')}")
    lines.append(f"  综合评价: {ratings.get('overall', 'N/A')}")

    # ASCII equity curve
    lines.append("\n【资金曲线 / Equity Curve】")
    lines.append(_ascii_chart(result.portfolio_values, width=56, height=15))

    # Trade markers timeline
    if result.trades:
        lines.append("\n【交易时间线 / Trade Timeline】")
        lines.append(_trade_timeline(result))

    lines.append("\n" + "=" * w)

    return "\n".join(lines)


def _ascii_chart(values: List[float], width: int = 56, height: int = 15) -> str:
    """Generate ASCII chart of values."""
    if not values:
        return "  (no data)"

    # Downsample if needed
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values

    min_val = min(sampled)
    max_val = max(sampled)
    val_range = max_val - min_val if max_val != min_val else 1

    lines = []
    for row in range(height - 1, -1, -1):
        threshold = min_val + (row / (height - 1)) * val_range
        line = "  "
        if row == height - 1:
            line += f"${max_val:>10,.0f} |"
        elif row == 0:
            line += f"${min_val:>10,.0f} |"
        elif row == height // 2:
            mid = (max_val + min_val) / 2
            line += f"${mid:>10,.0f} |"
        else:
            line += "            |"

        for val in sampled:
            normalized = (val - min_val) / val_range * (height - 1)
            if abs(normalized - row) < 0.5:
                line += "●"
            elif normalized > row:
                line += "│"
            else:
                line += " "
        lines.append(line)

    # X-axis
    lines.append("  " + "            +" + "─" * len(sampled))

    return "\n".join(lines)


def _trade_timeline(result: BacktestResult) -> str:
    """Show trades as a compact timeline."""
    lines = []
    for i, trade in enumerate(result.trades[:20]):  # Show max 20 trades
        arrow = "▲" if trade.pnl > 0 else "▼"
        color_indicator = "+" if trade.pnl > 0 else ""
        lines.append(
            f"  {arrow} #{i+1:2d}  {trade.entry_date} → {trade.exit_date}  "
            f"${trade.entry_price:.2f}→${trade.exit_price:.2f}  "
            f"{color_indicator}{trade.pnl_pct*100:.1f}%  "
            f"PnL: ${trade.pnl:+,.2f}  ({trade.holding_days}d)"
        )
    if len(result.trades) > 20:
        lines.append(f"  ... and {len(result.trades) - 20} more trades")
    return "\n".join(lines)


def compare_strategies(results: List[BacktestResult]) -> str:
    """
    对比多个策略 / Compare multiple strategies side by side.
    """
    if not results:
        return "No results to compare."

    lines = []
    lines.append("=" * 80)
    lines.append("  策略对比 / Strategy Comparison")
    lines.append("=" * 80)

    # Header
    header = f"{'指标 / Metric':<25s}"
    for r in results:
        short_name = r.strategy_name[:18]
        header += f" {short_name:>18s}"
    lines.append(header)
    lines.append("-" * 80)

    all_metrics = [calculate_all_metrics(r) for r in results]

    rows = [
        ("Total Return", "total_return_pct"),
        ("Annual Return", "annualized_return_pct"),
        ("Volatility", "annual_volatility_pct"),
        ("Max Drawdown", "max_drawdown_pct"),
        ("Sharpe Ratio", "sharpe_ratio"),
        ("Sortino Ratio", "sortino_ratio"),
        ("Calmar Ratio", "calmar_ratio"),
        ("Total Trades", "total_trades"),
        ("Win Rate", "win_rate_pct"),
        ("Profit Factor", "profit_factor"),
        ("Avg Holding Days", "avg_holding_days"),
        ("Commission", "total_commission"),
    ]

    for label, key in rows:
        row = f"  {label:<23s}"
        for m in all_metrics:
            val = m.get(key, "N/A")
            if isinstance(val, float):
                row += f" {val:>18.3f}"
            else:
                row += f" {str(val):>18s}"
        lines.append(row)

    # Winner determination
    lines.append("-" * 80)
    sharpes = [(m.get("sharpe_ratio", -999), i) for i, m in enumerate(all_metrics)]
    best_idx = max(sharpes, key=lambda x: x[0])[1]
    lines.append(f"  最佳策略 (按夏普比率): {results[best_idx].strategy_name}")

    lines.append("=" * 80)
    return "\n".join(lines)
