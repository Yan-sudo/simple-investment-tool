"""
策略评估指标 - Performance Metrics

这些指标用来检测你的策略好不好。
These metrics tell you if your strategy is any good.

【核心指标解读 / Key Metrics Explained】
- Sharpe Ratio (夏普比率):  >1 好, >2 很好, <0 亏损
- Max Drawdown (最大回撤):  <10% 优秀, <20% 可接受, >30% 危险
- Win Rate (胜率):          >50% 且盈亏比>1 才有意义
- Profit Factor (盈亏比):   >1.5 好, >2 很好
- Calmar Ratio:            年化收益/最大回撤, >1 好
"""

import math
from typing import List, Dict, Any
from engine.backtest import BacktestResult


def calculate_returns(values: List[float]) -> List[float]:
    """计算日收益率 / Calculate daily returns."""
    returns = []
    for i in range(1, len(values)):
        if values[i - 1] != 0:
            returns.append((values[i] - values[i - 1]) / values[i - 1])
        else:
            returns.append(0.0)
    return returns


def calculate_all_metrics(result: BacktestResult) -> Dict[str, Any]:
    """
    计算所有性能指标 / Calculate all performance metrics.

    Returns dict with all metrics and their interpretations.
    """
    metrics = {}
    pv = result.portfolio_values

    if not pv or len(pv) < 2:
        return {"error": "Insufficient data"}

    # --- 基础收益指标 / Basic Return Metrics ---
    total_return = (result.final_value - result.initial_capital) / result.initial_capital
    metrics["total_return"] = total_return
    metrics["total_return_pct"] = f"{total_return * 100:.2f}%"

    # Annualized return (assume 252 trading days)
    trading_days = len(pv)
    years = trading_days / 252
    if years > 0 and (1 + total_return) > 0:
        annualized_return = (1 + total_return) ** (1 / years) - 1
    else:
        annualized_return = 0.0
    metrics["annualized_return"] = annualized_return
    metrics["annualized_return_pct"] = f"{annualized_return * 100:.2f}%"

    # --- 风险指标 / Risk Metrics ---
    daily_returns = calculate_returns(pv)

    # Volatility (annualized)
    if daily_returns:
        avg_ret = sum(daily_returns) / len(daily_returns)
        variance = sum((r - avg_ret) ** 2 for r in daily_returns) / len(daily_returns)
        daily_vol = math.sqrt(variance)
        annual_vol = daily_vol * math.sqrt(252)
    else:
        daily_vol = annual_vol = 0.0
    metrics["annual_volatility"] = annual_vol
    metrics["annual_volatility_pct"] = f"{annual_vol * 100:.2f}%"

    # Max Drawdown
    peak = pv[0]
    max_dd = 0.0
    max_dd_start = max_dd_end = 0
    dd_start = 0
    for i in range(len(pv)):
        if pv[i] > peak:
            peak = pv[i]
            dd_start = i
        dd = (peak - pv[i]) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
            max_dd_start = dd_start
            max_dd_end = i
    metrics["max_drawdown"] = max_dd
    metrics["max_drawdown_pct"] = f"{max_dd * 100:.2f}%"
    if result.dates:
        metrics["max_drawdown_period"] = (
            f"{result.dates[max_dd_start]} to {result.dates[min(max_dd_end, len(result.dates)-1)]}"
        )

    # --- 风险调整收益 / Risk-Adjusted Returns ---
    # Sharpe Ratio (risk-free rate = 0.04)
    risk_free_daily = 0.04 / 252
    if daily_vol > 0:
        sharpe = (avg_ret - risk_free_daily) / daily_vol * math.sqrt(252)
    else:
        sharpe = 0.0
    metrics["sharpe_ratio"] = round(sharpe, 3)

    # Sortino Ratio (only downside volatility)
    downside_returns = [r for r in daily_returns if r < 0]
    if downside_returns:
        downside_var = sum(r ** 2 for r in downside_returns) / len(daily_returns)
        downside_vol = math.sqrt(downside_var)
        sortino = (avg_ret - risk_free_daily) / downside_vol * math.sqrt(252) if downside_vol > 0 else 0.0
    else:
        sortino = 0.0
    metrics["sortino_ratio"] = round(sortino, 3)

    # Calmar Ratio
    calmar = annualized_return / max_dd if max_dd > 0 else 0.0
    metrics["calmar_ratio"] = round(calmar, 3)

    # --- 交易统计 / Trade Statistics ---
    trades = result.trades
    metrics["total_trades"] = len(trades)

    if trades:
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]

        metrics["winning_trades"] = len(winning)
        metrics["losing_trades"] = len(losing)
        metrics["win_rate"] = len(winning) / len(trades)
        metrics["win_rate_pct"] = f"{len(winning) / len(trades) * 100:.1f}%"

        avg_win = sum(t.pnl for t in winning) / len(winning) if winning else 0
        avg_loss = abs(sum(t.pnl for t in losing) / len(losing)) if losing else 0
        metrics["avg_win"] = round(avg_win, 2)
        metrics["avg_loss"] = round(avg_loss, 2)

        # Profit Factor
        total_wins = sum(t.pnl for t in winning)
        total_losses = abs(sum(t.pnl for t in losing))
        metrics["profit_factor"] = round(total_wins / total_losses, 3) if total_losses > 0 else float('inf')

        # Average holding period
        avg_holding = sum(t.holding_days for t in trades) / len(trades)
        metrics["avg_holding_days"] = round(avg_holding, 1)

        # Best and worst trades
        best = max(trades, key=lambda t: t.pnl_pct)
        worst = min(trades, key=lambda t: t.pnl_pct)
        metrics["best_trade"] = f"{best.pnl_pct * 100:.2f}% ({best.entry_date})"
        metrics["worst_trade"] = f"{worst.pnl_pct * 100:.2f}% ({worst.entry_date})"

        # Max consecutive wins/losses
        max_consec_wins = max_consec_losses = 0
        cur_wins = cur_losses = 0
        for t in trades:
            if t.pnl > 0:
                cur_wins += 1
                cur_losses = 0
                max_consec_wins = max(max_consec_wins, cur_wins)
            else:
                cur_losses += 1
                cur_wins = 0
                max_consec_losses = max(max_consec_losses, cur_losses)
        metrics["max_consecutive_wins"] = max_consec_wins
        metrics["max_consecutive_losses"] = max_consec_losses
    else:
        metrics["winning_trades"] = 0
        metrics["losing_trades"] = 0
        metrics["win_rate_pct"] = "N/A"
        metrics["profit_factor"] = "N/A"

    metrics["total_commission"] = result.total_commission

    return metrics


def rate_strategy(metrics: Dict[str, Any]) -> Dict[str, str]:
    """
    给策略打分/评级 / Rate the strategy with simple grades.

    Returns dict of {metric_name: grade_and_comment}
    """
    ratings = {}

    # Sharpe Ratio
    sharpe = metrics.get("sharpe_ratio", 0)
    if sharpe >= 2.0:
        ratings["sharpe"] = "★★★ 优秀 (Excellent)"
    elif sharpe >= 1.0:
        ratings["sharpe"] = "★★☆ 良好 (Good)"
    elif sharpe >= 0.5:
        ratings["sharpe"] = "★☆☆ 一般 (Mediocre)"
    elif sharpe >= 0:
        ratings["sharpe"] = "☆☆☆ 差 (Poor)"
    else:
        ratings["sharpe"] = "✗ 亏损 (Losing money)"

    # Max Drawdown
    dd = metrics.get("max_drawdown", 1)
    if dd < 0.10:
        ratings["drawdown"] = "★★★ 优秀 (<10%)"
    elif dd < 0.20:
        ratings["drawdown"] = "★★☆ 可接受 (10-20%)"
    elif dd < 0.30:
        ratings["drawdown"] = "★☆☆ 较大风险 (20-30%)"
    else:
        ratings["drawdown"] = "✗ 危险 (>30%)"

    # Win Rate
    wr = metrics.get("win_rate", 0)
    pf = metrics.get("profit_factor", 0)
    if isinstance(pf, str):
        pf = 0
    if wr >= 0.6 and pf >= 1.5:
        ratings["trade_quality"] = "★★★ 高胜率+高盈亏比"
    elif wr >= 0.5 and pf >= 1.0:
        ratings["trade_quality"] = "★★☆ 合格"
    elif wr >= 0.4:
        ratings["trade_quality"] = "★☆☆ 需要更高盈亏比"
    else:
        ratings["trade_quality"] = "✗ 策略需要优化"

    # Overall
    score = 0
    if sharpe >= 1.0:
        score += 2
    elif sharpe >= 0.5:
        score += 1
    if dd < 0.20:
        score += 2
    elif dd < 0.30:
        score += 1
    if wr >= 0.5 and (isinstance(pf, (int, float)) and pf >= 1.0):
        score += 2
    elif wr >= 0.4:
        score += 1

    if score >= 5:
        ratings["overall"] = "★★★ 策略表现优秀，可进入模拟盘测试"
    elif score >= 3:
        ratings["overall"] = "★★☆ 有潜力，建议调参优化"
    else:
        ratings["overall"] = "★☆☆ 需要重大改进或更换策略"

    return ratings
