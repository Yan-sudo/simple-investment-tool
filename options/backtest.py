"""
Option Backtesting Engine

Extends the existing stock backtesting engine to handle option positions:
- Multi-leg option trades
- Time decay simulation
- Position tracking with Greeks
- Comprehensive performance reporting
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from options.strategies import (
    OptionStrategy, OptionPosition, OptionTrade,
    IronCondorStrategy, VerticalSpreadStrategy, WheelStrategy,
    StraddleStrategy, IVAdaptiveStrategy,
)
from options.iv_model import (
    historical_volatility, generate_iv_series, iv_rank, iv_percentile,
    classify_vol_regime,
)
from options.pricing import call_price, put_price, all_greeks
from data.market_data import MarketData


@dataclass
class OptionTradeRecord:
    """Record of a completed option trade."""
    strategy_name: str
    entry_date: str
    exit_date: str
    underlying_entry: float
    underlying_exit: float
    legs_description: str
    net_credit: float       # Initial credit/debit
    exit_pnl: float         # Realized P&L
    pnl_pct: float          # P&L as % of max risk
    max_risk: float          # Capital at risk
    holding_days: int
    iv_at_entry: float
    iv_rank_at_entry: float


@dataclass
class OptionBacktestResult:
    """Complete results from an option strategy backtest."""
    strategy_name: str
    params: Dict
    symbol: str

    # Time series
    dates: List[str]
    portfolio_values: List[float]
    cash_history: List[float]

    # Trades
    trades: List[OptionTradeRecord]
    positions_opened: int
    positions_closed: int

    # Capital
    initial_capital: float
    final_value: float

    # Summary
    total_pnl: float
    total_return_pct: float
    annualized_return_pct: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_holding_days: float
    total_premium_collected: float
    total_premium_paid: float


class OptionBacktestEngine:
    """Engine for backtesting option strategies."""

    def __init__(self, initial_capital: float = 100_000.0,
                 commission_per_contract: float = 0.65,
                 max_position_pct: float = 0.20):
        """
        Args:
            initial_capital: Starting cash
            commission_per_contract: Commission per contract per leg
            max_position_pct: Max capital allocation per position (20%)
        """
        self.initial_capital = initial_capital
        self.commission_per_contract = commission_per_contract
        self.max_position_pct = max_position_pct

    def run(self, strategy: OptionStrategy, data: MarketData) -> OptionBacktestResult:
        """Run a full option strategy backtest.

        Args:
            strategy: OptionStrategy instance
            data: MarketData with OHLCV data

        Returns:
            OptionBacktestResult with complete performance data
        """
        closes = data.close
        highs = data.high
        lows = data.low
        dates = [str(d) for d in data.dates]
        n = len(closes)

        # Generate strategy trades
        positions = strategy.generate_trades(closes, highs, lows, dates)

        # Track portfolio
        cash = self.initial_capital
        portfolio_values = [self.initial_capital] * n
        cash_history = [self.initial_capital] * n
        trades = []
        total_premium_collected = 0.0
        total_premium_paid = 0.0
        total_commission = 0.0

        # IV data for reporting
        iv_series = generate_iv_series(closes, base_iv=0.20, seed=42)

        active_positions: List[Tuple[int, OptionPosition]] = []  # (entry_idx, position)

        for i in range(n):
            # Check for new positions
            if positions[i] is not None:
                pos = positions[i]
                max_risk = pos.max_loss if pos.max_loss < float('inf') else \
                    self.initial_capital * self.max_position_pct

                # Check if we have enough capital
                if max_risk <= cash * 0.95:
                    # Commission for opening
                    num_legs = len(pos.legs)
                    num_contracts = sum(l.contracts for l in pos.legs)
                    commission = self.commission_per_contract * num_contracts
                    total_commission += commission

                    # Update cash with premium and commission
                    credit = pos.net_credit
                    cash += credit - commission

                    if credit > 0:
                        total_premium_collected += credit
                    else:
                        total_premium_paid += abs(credit)

                    active_positions.append((i, pos))

            # Check for exiting positions
            closed = []
            for idx, (entry_idx, pos) in enumerate(active_positions):
                if pos.exit_date == dates[i]:
                    # Close position
                    pnl = pos.exit_pnl

                    # Commission for closing
                    num_contracts = sum(l.contracts for l in pos.legs)
                    commission = self.commission_per_contract * num_contracts
                    total_commission += commission

                    cash += pnl - commission

                    # Calculate max risk for P&L %
                    max_risk = pos.max_loss if 0 < pos.max_loss < float('inf') else \
                        abs(pos.net_credit) if pos.net_credit < 0 else \
                        pos.net_credit * 2

                    holding_days = i - entry_idx

                    # IV rank at entry
                    entry_iv = iv_series[entry_idx] if entry_idx < len(iv_series) else 0.2
                    iv_history = iv_series[max(0, entry_idx - 252):entry_idx]
                    ivr_entry = iv_rank(entry_iv, iv_history) if iv_history else 50.0

                    trade_record = OptionTradeRecord(
                        strategy_name=pos.strategy_name,
                        entry_date=pos.entry_date,
                        exit_date=dates[i],
                        underlying_entry=pos.underlying_price,
                        underlying_exit=closes[i],
                        legs_description=self._describe_legs(pos),
                        net_credit=pos.net_credit,
                        exit_pnl=pnl,
                        pnl_pct=(pnl / max_risk * 100) if max_risk > 0 else 0,
                        max_risk=max_risk,
                        holding_days=holding_days,
                        iv_at_entry=entry_iv,
                        iv_rank_at_entry=ivr_entry,
                    )
                    trades.append(trade_record)
                    closed.append(idx)

            # Remove closed positions (reverse order)
            for idx in sorted(closed, reverse=True):
                active_positions.pop(idx)

            # Track portfolio value
            portfolio_values[i] = cash
            cash_history[i] = cash

        # Calculate metrics
        metrics = self._calculate_metrics(
            portfolio_values, trades, dates, self.initial_capital
        )

        return OptionBacktestResult(
            strategy_name=strategy.name,
            params=strategy.params,
            symbol=data.dates[0].__class__.__name__ if data.dates else "SYNTHETIC",
            dates=dates,
            portfolio_values=portfolio_values,
            cash_history=cash_history,
            trades=trades,
            positions_opened=sum(1 for p in positions if p is not None),
            positions_closed=len(trades),
            initial_capital=self.initial_capital,
            final_value=portfolio_values[-1],
            **metrics,
        )

    def _describe_legs(self, pos: OptionPosition) -> str:
        parts = []
        for leg in pos.legs:
            parts.append(f"{leg.action[0].upper()}{leg.contracts} "
                         f"{leg.strike:.0f}{leg.option_type[0].upper()} "
                         f"@{leg.premium:.2f}")
        return " / ".join(parts)

    def _calculate_metrics(self, portfolio_values, trades, dates,
                           initial_capital) -> Dict:
        """Calculate all performance metrics."""
        final_value = portfolio_values[-1]
        total_pnl = final_value - initial_capital
        total_return_pct = (total_pnl / initial_capital) * 100

        # Annualized return
        n_days = len(dates)
        years = n_days / 252.0
        if years > 0 and final_value > 0:
            annualized_return = (final_value / initial_capital) ** (1 / years) - 1
        else:
            annualized_return = 0.0

        # Win/loss stats
        wins = [t for t in trades if t.exit_pnl > 0]
        losses = [t for t in trades if t.exit_pnl <= 0]
        win_rate = len(wins) / len(trades) * 100 if trades else 0
        avg_win = sum(t.exit_pnl for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.exit_pnl for t in losses) / len(losses) if losses else 0

        total_wins = sum(t.exit_pnl for t in wins)
        total_losses = abs(sum(t.exit_pnl for t in losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Max drawdown
        peak = portfolio_values[0]
        max_dd = 0.0
        for v in portfolio_values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Sharpe ratio
        if len(portfolio_values) > 1:
            daily_returns = []
            for i in range(1, len(portfolio_values)):
                if portfolio_values[i - 1] > 0:
                    daily_returns.append(
                        (portfolio_values[i] - portfolio_values[i - 1]) / portfolio_values[i - 1]
                    )
            if daily_returns:
                mean_ret = sum(daily_returns) / len(daily_returns)
                std_ret = math.sqrt(
                    sum((r - mean_ret) ** 2 for r in daily_returns) / max(len(daily_returns) - 1, 1)
                )
                risk_free_daily = 0.04 / 252
                sharpe = (mean_ret - risk_free_daily) / std_ret * math.sqrt(252) if std_ret > 0 else 0
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        # Average holding days
        avg_holding = (sum(t.holding_days for t in trades) / len(trades)) if trades else 0

        # Premium stats
        total_collected = sum(t.net_credit for t in trades if t.net_credit > 0)
        total_paid = sum(abs(t.net_credit) for t in trades if t.net_credit < 0)

        return {
            "total_pnl": total_pnl,
            "total_return_pct": total_return_pct,
            "annualized_return_pct": annualized_return * 100,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_drawdown_pct": max_dd * 100,
            "sharpe_ratio": sharpe,
            "avg_holding_days": avg_holding,
            "total_premium_collected": total_collected,
            "total_premium_paid": total_paid,
        }


# ── Report Generation ───────────────────────────────────────────────────────

def generate_option_report(result: OptionBacktestResult) -> str:
    """Generate a comprehensive text report for option backtest results."""
    lines = []
    sep = "=" * 70

    lines.append(sep)
    lines.append(f"  OPTION STRATEGY BACKTEST REPORT")
    lines.append(sep)
    lines.append("")

    # Strategy info
    lines.append(f"  Strategy:        {result.strategy_name}")
    lines.append(f"  Parameters:      {result.params}")
    lines.append(f"  Period:          {result.dates[0]} → {result.dates[-1]}")
    lines.append(f"  Trading Days:    {len(result.dates)}")
    lines.append("")

    # Returns
    lines.append("─" * 70)
    lines.append("  RETURNS")
    lines.append("─" * 70)
    lines.append(f"  Initial Capital: ${result.initial_capital:>12,.2f}")
    lines.append(f"  Final Value:     ${result.final_value:>12,.2f}")
    lines.append(f"  Total P&L:       ${result.total_pnl:>+12,.2f}")
    lines.append(f"  Total Return:    {result.total_return_pct:>+12.2f}%")
    lines.append(f"  Annualized:      {result.annualized_return_pct:>+12.2f}%")
    lines.append("")

    # Risk metrics
    lines.append("─" * 70)
    lines.append("  RISK METRICS")
    lines.append("─" * 70)
    lines.append(f"  Max Drawdown:    {result.max_drawdown_pct:>12.2f}%")
    lines.append(f"  Sharpe Ratio:    {result.sharpe_ratio:>12.2f}")
    lines.append("")

    # Trade stats
    lines.append("─" * 70)
    lines.append("  TRADE STATISTICS")
    lines.append("─" * 70)
    lines.append(f"  Positions Opened:{result.positions_opened:>12d}")
    lines.append(f"  Positions Closed:{result.positions_closed:>12d}")
    lines.append(f"  Win Rate:        {result.win_rate:>12.1f}%")
    lines.append(f"  Avg Win:         ${result.avg_win:>+12,.2f}")
    lines.append(f"  Avg Loss:        ${result.avg_loss:>+12,.2f}")
    lines.append(f"  Profit Factor:   {result.profit_factor:>12.2f}")
    lines.append(f"  Avg Hold (days): {result.avg_holding_days:>12.1f}")
    lines.append("")

    # Premium analysis
    lines.append("─" * 70)
    lines.append("  PREMIUM ANALYSIS")
    lines.append("─" * 70)
    lines.append(f"  Premium Collected: ${result.total_premium_collected:>10,.2f}")
    lines.append(f"  Premium Paid:      ${result.total_premium_paid:>10,.2f}")
    net_prem = result.total_premium_collected - result.total_premium_paid
    lines.append(f"  Net Premium:       ${net_prem:>+10,.2f}")
    lines.append("")

    # Strategy rating
    lines.append("─" * 70)
    lines.append("  STRATEGY RATING")
    lines.append("─" * 70)
    lines.append(f"  Sharpe:     {_rate_sharpe(result.sharpe_ratio)}")
    lines.append(f"  Drawdown:   {_rate_drawdown(result.max_drawdown_pct)}")
    lines.append(f"  Win Rate:   {_rate_winrate(result.win_rate)}")
    lines.append(f"  Overall:    {_rate_overall(result)}")
    lines.append("")

    # Equity curve (ASCII)
    lines.append("─" * 70)
    lines.append("  EQUITY CURVE")
    lines.append("─" * 70)
    lines.append(_ascii_chart(result.portfolio_values, width=60, height=15))
    lines.append("")

    # Trade log (last 20)
    lines.append("─" * 70)
    lines.append("  RECENT TRADES (last 20)")
    lines.append("─" * 70)
    recent = result.trades[-20:] if len(result.trades) > 20 else result.trades
    if recent:
        lines.append(f"  {'Entry':>10} → {'Exit':>10}  {'Type':<18} {'P&L':>10} {'%':>7} {'Days':>5}")
        lines.append("  " + "-" * 66)
        for t in recent:
            arrow = "▲" if t.exit_pnl > 0 else "▼"
            lines.append(
                f"  {t.entry_date:>10} → {t.exit_date:>10}  "
                f"{t.strategy_name:<18} "
                f"${t.exit_pnl:>+9,.0f} {t.pnl_pct:>+6.1f}% {t.holding_days:>4}d {arrow}"
            )
    else:
        lines.append("  No completed trades.")
    lines.append("")

    lines.append(sep)
    return "\n".join(lines)


def compare_option_strategies(results: List[OptionBacktestResult]) -> str:
    """Compare multiple option strategies side by side."""
    lines = []
    sep = "=" * 90

    lines.append(sep)
    lines.append("  OPTION STRATEGY COMPARISON")
    lines.append(sep)
    lines.append("")

    # Header
    header = f"  {'Metric':<25}"
    for r in results:
        header += f" {r.strategy_name:>15}"
    lines.append(header)
    lines.append("  " + "-" * (25 + 16 * len(results)))

    # Metrics
    metrics = [
        ("Total Return %", [f"{r.total_return_pct:+.2f}%" for r in results]),
        ("Annualized Return %", [f"{r.annualized_return_pct:+.2f}%" for r in results]),
        ("Sharpe Ratio", [f"{r.sharpe_ratio:.2f}" for r in results]),
        ("Max Drawdown %", [f"{r.max_drawdown_pct:.2f}%" for r in results]),
        ("Win Rate %", [f"{r.win_rate:.1f}%" for r in results]),
        ("Profit Factor", [f"{r.profit_factor:.2f}" for r in results]),
        ("Total Trades", [f"{r.positions_closed}" for r in results]),
        ("Avg Win $", [f"${r.avg_win:+,.0f}" for r in results]),
        ("Avg Loss $", [f"${r.avg_loss:+,.0f}" for r in results]),
        ("Avg Hold Days", [f"{r.avg_holding_days:.0f}" for r in results]),
        ("Final Value $", [f"${r.final_value:,.0f}" for r in results]),
    ]

    for name, values in metrics:
        row = f"  {name:<25}"
        for v in values:
            row += f" {v:>15}"
        lines.append(row)

    lines.append("")

    # Winner
    best = max(results, key=lambda r: r.sharpe_ratio)
    lines.append(f"  Best Strategy (by Sharpe): {best.strategy_name} "
                 f"(Sharpe={best.sharpe_ratio:.2f})")
    lines.append("")
    lines.append(sep)

    return "\n".join(lines)


def _rate_sharpe(sharpe):
    if sharpe > 2.0:
        return "★★★ Excellent"
    elif sharpe > 1.0:
        return "★★☆ Good"
    elif sharpe > 0.5:
        return "★☆☆ Acceptable"
    elif sharpe > 0:
        return "☆☆☆ Weak"
    else:
        return "✗ Negative risk-adjusted return"


def _rate_drawdown(dd):
    if dd < 5:
        return "★★★ Very Low Risk"
    elif dd < 10:
        return "★★☆ Moderate Risk"
    elif dd < 20:
        return "★☆☆ Elevated Risk"
    else:
        return "✗ High Risk"


def _rate_winrate(wr):
    if wr > 70:
        return "★★★ High"
    elif wr > 55:
        return "★★☆ Above Average"
    elif wr > 40:
        return "★☆☆ Average"
    else:
        return "☆☆☆ Below Average"


def _rate_overall(result):
    score = 0
    if result.sharpe_ratio > 1.0:
        score += 2
    elif result.sharpe_ratio > 0.5:
        score += 1
    if result.max_drawdown_pct < 10:
        score += 2
    elif result.max_drawdown_pct < 20:
        score += 1
    if result.win_rate > 60:
        score += 2
    elif result.win_rate > 45:
        score += 1

    if score >= 5:
        return "★★★ Strong Strategy"
    elif score >= 3:
        return "★★☆ Decent Strategy"
    elif score >= 1:
        return "★☆☆ Needs Improvement"
    else:
        return "☆☆☆ Poor Performance"


def _ascii_chart(values, width=60, height=15):
    """Simple ASCII equity curve chart."""
    if not values:
        return "  (no data)"

    n = len(values)
    if n > width:
        step = n / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values

    v_min = min(sampled)
    v_max = max(sampled)
    v_range = v_max - v_min if v_max != v_min else 1.0

    lines = []
    for row in range(height, -1, -1):
        threshold = v_min + (row / height) * v_range
        line = "  "
        if row == height:
            line += f"${v_max:>10,.0f} │"
        elif row == 0:
            line += f"${v_min:>10,.0f} │"
        elif row == height // 2:
            mid = (v_max + v_min) / 2
            line += f"${mid:>10,.0f} │"
        else:
            line += "            │"

        for v in sampled:
            level = (v - v_min) / v_range * height
            if abs(level - row) < 0.5:
                line += "●"
            elif level > row:
                line += "│" if row > 0 else "─"
            else:
                line += " "
        lines.append(line)

    # X-axis
    lines.append("  " + "            └" + "─" * len(sampled))

    return "\n".join(lines)
