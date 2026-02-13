"""
Option Backtesting Engine (v2 — Enhanced)

Upgrades over v1:
- Realistic slippage: 10% of bid-ask spread per leg (not flat %)
- EV gating: only enter trades with positive expected value
- Greeks monitoring: track portfolio Greeks and generate alerts
- Wash sale detection: flag tax compliance issues
- Dynamic hedging: track delta exposure and suggest hedges
- Enhanced reporting with risk analysis sections
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from options.strategies import (
    OptionStrategy, OptionPosition, OptionTrade,
    IronCondorStrategy, VerticalSpreadStrategy, WheelStrategy,
    StraddleStrategy, IVAdaptiveStrategy,
    ButterflySpreadStrategy, PCREnhancedStrategy,
)
from options.iv_model import (
    historical_volatility, generate_iv_series, iv_rank, iv_percentile,
    classify_vol_regime,
)
from options.pricing import call_price, put_price, all_greeks
from options.risk_manager import GreeksSentinel, PortfolioRisk, format_risk_report
from options.ev_model import EVAnalyzer, EVResult, format_ev_report
from options.tax_compliance import WashSaleDetector, format_wash_sale_report
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
    max_risk: float         # Capital at risk
    holding_days: int
    iv_at_entry: float
    iv_rank_at_entry: float
    # v2 additions
    hv_at_entry: float = 0.0
    ev_at_entry: float = 0.0      # Expected value when entering
    pop_at_entry: float = 0.0     # Probability of profit
    slippage_cost: float = 0.0    # Slippage paid
    commission_cost: float = 0.0  # Commission paid
    greeks_alerts: int = 0        # Number of Greeks alerts during hold
    exit_reason: str = ""         # Why the trade was closed


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
    sortino_ratio: float
    avg_holding_days: float
    total_premium_collected: float
    total_premium_paid: float

    # v2 additions
    total_slippage: float = 0.0
    total_commission: float = 0.0
    total_transaction_costs: float = 0.0
    avg_ev_at_entry: float = 0.0
    avg_pop_at_entry: float = 0.0
    ev_filtered_trades: int = 0       # Trades rejected by EV filter
    total_greeks_alerts: int = 0
    wash_sale_violations: int = 0
    wash_sale_disallowed: float = 0.0
    risk_snapshot: Optional[PortfolioRisk] = None
    # v3 additions
    calmar_ratio: float = 0.0         # Annualized return / max drawdown
    max_consecutive_losses: int = 0   # Longest losing streak


class OptionBacktestEngine:
    """Enhanced engine for backtesting option strategies.

    v2 Features:
    - Realistic slippage: 10% of mid-price as spread cost per leg
    - EV gating: optionally filter out negative-EV trades
    - Greeks monitoring: track and alert on Greek thresholds
    - Wash sale detection: track 30-day tax rule violations
    """

    def __init__(self, initial_capital: float = 100_000.0,
                 commission_per_contract: float = 0.65,
                 slippage_pct_of_spread: float = 0.10,
                 max_position_pct: float = 0.20,
                 enable_ev_filter: bool = True,
                 enable_greeks_monitor: bool = True,
                 enable_wash_sale: bool = True,
                 delta_threshold: float = 0.50,
                 gamma_dollar_threshold: float = 500.0):
        """
        Args:
            initial_capital: Starting cash
            commission_per_contract: Commission per contract per leg ($0.65)
            slippage_pct_of_spread: Slippage as % of bid-ask spread (10%)
            max_position_pct: Max capital per position (20%)
            enable_ev_filter: If True, only enter +EV trades
            enable_greeks_monitor: If True, track Greeks and generate alerts
            enable_wash_sale: If True, detect wash sale violations
            delta_threshold: Greeks sentinel delta alert threshold
            gamma_dollar_threshold: Greeks sentinel gamma dollar threshold
        """
        self.initial_capital = initial_capital
        self.commission_per_contract = commission_per_contract
        self.slippage_pct = slippage_pct_of_spread
        self.max_position_pct = max_position_pct
        self.enable_ev_filter = enable_ev_filter
        self.enable_greeks_monitor = enable_greeks_monitor
        self.enable_wash_sale = enable_wash_sale

        # Initialize sub-modules
        self.ev_analyzer = EVAnalyzer(
            commission_per_contract=commission_per_contract,
            slippage_pct=slippage_pct_of_spread,
        )
        self.sentinel = GreeksSentinel(
            delta_threshold=delta_threshold,
            gamma_dollar_threshold=gamma_dollar_threshold,
            portfolio_capital=initial_capital,
        )
        self.wash_detector = WashSaleDetector(conservative=True)

    def _calculate_slippage(self, legs, underlying_price: float,
                            iv: float, T: float, r: float) -> float:
        """Calculate realistic slippage based on bid-ask spread.

        Slippage model:
            For each leg:
                theoretical_price = BS_price(S, K, T, r, iv)
                estimated_spread = price × spread_factor
                    spread_factor = 0.02 (ATM) to 0.15 (deep OTM)
                slippage_per_leg = spread × slippage_pct × contracts × 100
        """
        total_slippage = 0.0
        for leg in legs:
            if leg.option_type == "call":
                price = call_price(underlying_price, leg.strike, T, r, iv)
            else:
                price = put_price(underlying_price, leg.strike, T, r, iv)

            # Estimate spread: wider for OTM options
            moneyness = abs(leg.strike / underlying_price - 1.0)
            spread_factor = 0.02 + moneyness * 0.5  # 2% ATM to ~15%+ deep OTM
            spread_factor = min(spread_factor, 0.30)
            estimated_spread = max(0.01, price * spread_factor)

            slippage = estimated_spread * self.slippage_pct * leg.contracts * 100
            total_slippage += slippage

        return total_slippage

    def _evaluate_ev(self, pos: OptionPosition, underlying_price: float,
                     iv: float, hv: float, T: float) -> Optional[EVResult]:
        """Evaluate expected value of a position before entering."""
        S = underlying_price

        if pos.strategy_name == "iron_condor" and len(pos.legs) == 4:
            strikes = sorted([l.strike for l in pos.legs])
            put_legs = sorted([l for l in pos.legs if l.option_type == "put"],
                              key=lambda l: l.strike, reverse=True)
            call_legs = sorted([l for l in pos.legs if l.option_type == "call"],
                               key=lambda l: l.strike)
            if len(put_legs) >= 2 and len(call_legs) >= 2:
                return self.ev_analyzer.analyze_iron_condor(
                    S, put_legs[0].strike, put_legs[1].strike,
                    call_legs[0].strike, call_legs[1].strike,
                    T, iv, hv
                )

        elif pos.strategy_name in ("bull_put_spread", "bear_call_spread"):
            legs_sorted = sorted(pos.legs, key=lambda l: l.strike, reverse=True)
            spread_type = "put" if pos.strategy_name == "bull_put_spread" else "call"
            if spread_type == "put":
                return self.ev_analyzer.analyze_credit_spread(
                    S, legs_sorted[0].strike, legs_sorted[1].strike,
                    T, iv, hv, "put"
                )
            else:
                return self.ev_analyzer.analyze_credit_spread(
                    S, legs_sorted[1].strike, legs_sorted[0].strike,
                    T, iv, hv, "call"
                )

        elif pos.strategy_name in ("long_straddle", "long_strangle"):
            call_leg = [l for l in pos.legs if l.option_type == "call"][0]
            put_leg = [l for l in pos.legs if l.option_type == "put"][0]
            return self.ev_analyzer.analyze_long_straddle(
                S, call_leg.strike, put_leg.strike, T, iv, hv
            )

        elif pos.strategy_name == "pcr_enhanced_ic" and len(pos.legs) == 4:
            # Same EV logic as regular IC
            put_legs = sorted([l for l in pos.legs if l.option_type == "put"],
                              key=lambda l: l.strike, reverse=True)
            call_legs = sorted([l for l in pos.legs if l.option_type == "call"],
                               key=lambda l: l.strike)
            if len(put_legs) >= 2 and len(call_legs) >= 2:
                return self.ev_analyzer.analyze_iron_condor(
                    S, put_legs[0].strike, put_legs[1].strike,
                    call_legs[0].strike, call_legs[1].strike,
                    T, iv, hv
                )

        # For butterfly, TA-driven debit strategies, and others: skip EV filter
        # (debit strategies are not well modelled by the credit EV framework)
        return None

    def run(self, strategy: OptionStrategy, data: MarketData) -> OptionBacktestResult:
        """Run a full option strategy backtest with all v2 enhancements."""
        closes = data.close
        highs = data.high
        lows = data.low
        dates = [str(d) for d in data.dates]
        n = len(closes)

        # Generate strategy trades
        positions = strategy.generate_trades(closes, highs, lows, dates)

        # Pre-compute volatility data
        iv_series = generate_iv_series(closes, base_iv=0.20, seed=42)
        hv_series = historical_volatility(closes, window=20)

        # Track portfolio
        cash = self.initial_capital
        portfolio_values = [self.initial_capital] * n
        cash_history = [self.initial_capital] * n
        trades = []
        total_premium_collected = 0.0
        total_premium_paid = 0.0
        total_commission = 0.0
        total_slippage = 0.0
        ev_filtered_count = 0
        total_greeks_alerts = 0

        # Reset wash sale detector
        self.wash_detector = WashSaleDetector(conservative=True)

        active_positions: List[Tuple[int, OptionPosition, float, float, EVResult]] = []
        # (entry_idx, position, commission_paid, slippage_paid, ev_result)

        for i in range(n):
            current_iv = iv_series[i] if i < len(iv_series) else 0.20
            current_hv = hv_series[i] if i < len(hv_series) else 0.20

            # ── Check for new positions ──
            if positions[i] is not None:
                pos = positions[i]
                max_risk = pos.max_loss if pos.max_loss < float('inf') else \
                    self.initial_capital * self.max_position_pct

                # Check if we have enough capital
                if max_risk <= cash * 0.95:
                    T = pos.legs[0].dte / 365.0 if pos.legs else 30 / 365.0
                    r = 0.04

                    # EV filter
                    ev_result = None
                    if self.enable_ev_filter:
                        ev_result = self._evaluate_ev(
                            pos, closes[i], current_iv, current_hv, T
                        )
                        if ev_result is not None and not ev_result.is_positive_ev:
                            # Skip negative EV trades
                            ev_filtered_count += 1
                            positions[i] = None
                            portfolio_values[i] = cash
                            cash_history[i] = cash
                            continue

                    # Wash sale check
                    if self.enable_wash_sale:
                        ws_alerts = self.wash_detector.check_new_trade(
                            underlying=data.dates[0].__class__.__name__
                            if data.dates else "SPY",
                            option_type="option",
                            strike=None,
                            trade_date=dates[i],
                        )
                        # Don't block, just record alerts

                    # Calculate realistic costs
                    num_contracts = sum(l.contracts for l in pos.legs)
                    commission = self.commission_per_contract * num_contracts
                    slippage = self._calculate_slippage(
                        pos.legs, closes[i], current_iv, T, r
                    )

                    total_commission += commission * 2  # open + close estimated
                    total_slippage += slippage

                    # Update cash
                    credit = pos.net_credit
                    cash += credit - commission - slippage

                    if credit > 0:
                        total_premium_collected += credit
                    else:
                        total_premium_paid += abs(credit)

                    active_positions.append((i, pos, commission, slippage,
                                             ev_result))

            # ── Check for exiting positions ──
            closed = []
            for idx, (entry_idx, pos, entry_comm, entry_slip, ev_result) in enumerate(active_positions):
                if pos.exit_date == dates[i]:
                    pnl = pos.exit_pnl

                    # Closing costs
                    num_contracts = sum(l.contracts for l in pos.legs)
                    close_commission = self.commission_per_contract * num_contracts
                    T_remain = max(1 / 365.0, (pos.legs[0].dte - (i - entry_idx)) / 365.0)
                    close_slippage = self._calculate_slippage(
                        pos.legs, closes[i], current_iv, T_remain, 0.04
                    )

                    total_commission += close_commission
                    total_slippage += close_slippage

                    cash += pnl - close_commission - close_slippage

                    # Max risk for P&L %
                    max_risk = pos.max_loss if 0 < pos.max_loss < float('inf') else \
                        abs(pos.net_credit) if pos.net_credit < 0 else \
                        pos.net_credit * 2
                    holding_days = i - entry_idx

                    # IV/HV at entry
                    entry_iv = iv_series[entry_idx] if entry_idx < len(iv_series) else 0.2
                    entry_hv = hv_series[entry_idx] if entry_idx < len(hv_series) else 0.2
                    iv_history = iv_series[max(0, entry_idx - 252):entry_idx]
                    ivr_entry = iv_rank(entry_iv, iv_history) if iv_history else 50.0

                    # Greeks alerts during hold
                    greeks_alerts_count = 0
                    if self.enable_greeks_monitor:
                        _, alerts = self.sentinel.check_position(
                            pos.legs, closes[i],
                            max(1, pos.legs[0].dte - holding_days),
                            current_iv, pos.net_credit
                        )
                        greeks_alerts_count = len(alerts)
                        total_greeks_alerts += greeks_alerts_count

                    # Determine exit reason
                    exit_reason = "DTE_EXIT"
                    if hasattr(pos, '_exit_reason'):
                        exit_reason = pos._exit_reason
                    remaining_dte = pos.legs[0].dte - holding_days if pos.legs else 0
                    if remaining_dte <= 0:
                        exit_reason = "EXPIRATION"

                    total_costs = entry_comm + entry_slip + close_commission + close_slippage

                    trade_record = OptionTradeRecord(
                        strategy_name=pos.strategy_name,
                        entry_date=pos.entry_date,
                        exit_date=dates[i],
                        underlying_entry=pos.underlying_price,
                        underlying_exit=closes[i],
                        legs_description=self._describe_legs(pos),
                        net_credit=pos.net_credit,
                        exit_pnl=pnl - total_costs,  # Net of costs
                        pnl_pct=((pnl - total_costs) / max_risk * 100) if max_risk > 0 else 0,
                        max_risk=max_risk,
                        holding_days=holding_days,
                        iv_at_entry=entry_iv,
                        iv_rank_at_entry=ivr_entry,
                        hv_at_entry=entry_hv,
                        ev_at_entry=ev_result.expected_value if ev_result else 0,
                        pop_at_entry=ev_result.pop if ev_result else 0,
                        slippage_cost=entry_slip + close_slippage,
                        commission_cost=entry_comm + close_commission,
                        greeks_alerts=greeks_alerts_count,
                        exit_reason=exit_reason,
                    )
                    trades.append(trade_record)
                    closed.append(idx)

                    # Record for wash sale tracking
                    if self.enable_wash_sale:
                        self.wash_detector.record_close(
                            underlying="SPY",
                            option_type="option",
                            strike=None,
                            entry_date=pos.entry_date,
                            exit_date=dates[i],
                            pnl=pnl - total_costs,
                        )

            for idx in sorted(closed, reverse=True):
                active_positions.pop(idx)

            portfolio_values[i] = cash
            cash_history[i] = cash

        # Calculate metrics
        metrics = self._calculate_metrics(
            portfolio_values, trades, dates, self.initial_capital
        )

        # Final risk snapshot
        risk_snapshot = None
        if self.enable_greeks_monitor and active_positions:
            risk_positions = []
            for entry_idx, pos, _, _, _ in active_positions:
                remaining = max(1, pos.legs[0].dte - (n - 1 - entry_idx))
                risk_positions.append((remaining, pos.legs, pos.net_credit))
            risk_snapshot = self.sentinel.check_portfolio(
                risk_positions, closes[-1], iv_series[-1]
            )

        # Wash sale summary
        ws_summary = self.wash_detector.get_summary()

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
            total_slippage=total_slippage,
            total_commission=total_commission,
            total_transaction_costs=total_slippage + total_commission,
            avg_ev_at_entry=(sum(t.ev_at_entry for t in trades) / len(trades))
                if trades else 0,
            avg_pop_at_entry=(sum(t.pop_at_entry for t in trades) / len(trades))
                if trades else 0,
            ev_filtered_trades=ev_filtered_count,
            total_greeks_alerts=total_greeks_alerts,
            wash_sale_violations=ws_summary['wash_sale_violations'],
            wash_sale_disallowed=ws_summary['total_disallowed_losses'],
            risk_snapshot=risk_snapshot,
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
        """Calculate all performance metrics including Sortino ratio."""
        final_value = portfolio_values[-1]
        total_pnl = final_value - initial_capital
        total_return_pct = (total_pnl / initial_capital) * 100

        n_days = len(dates)
        years = n_days / 252.0
        if years > 0 and final_value > 0:
            annualized_return = (final_value / initial_capital) ** (1 / years) - 1
        else:
            annualized_return = 0.0

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

        # Daily returns
        daily_returns = []
        if len(portfolio_values) > 1:
            for i in range(1, len(portfolio_values)):
                if portfolio_values[i - 1] > 0:
                    daily_returns.append(
                        (portfolio_values[i] - portfolio_values[i - 1]) / portfolio_values[i - 1]
                    )

        # Sharpe ratio
        if daily_returns:
            mean_ret = sum(daily_returns) / len(daily_returns)
            std_ret = math.sqrt(
                sum((r - mean_ret) ** 2 for r in daily_returns) / max(len(daily_returns) - 1, 1)
            )
            risk_free_daily = 0.04 / 252
            sharpe = (mean_ret - risk_free_daily) / std_ret * math.sqrt(252) if std_ret > 0 else 0
        else:
            sharpe = 0.0

        # Sortino ratio (uses only downside deviation)
        if daily_returns:
            mean_ret = sum(daily_returns) / len(daily_returns)
            risk_free_daily = 0.04 / 252
            downside_returns = [min(r - risk_free_daily, 0) for r in daily_returns]
            downside_var = sum(r ** 2 for r in downside_returns) / max(len(downside_returns) - 1, 1)
            downside_dev = math.sqrt(downside_var)
            sortino = (mean_ret - risk_free_daily) / downside_dev * math.sqrt(252) if downside_dev > 0 else 0
        else:
            sortino = 0.0

        avg_holding = (sum(t.holding_days for t in trades) / len(trades)) if trades else 0
        total_collected = sum(t.net_credit for t in trades if t.net_credit > 0)
        total_paid = sum(abs(t.net_credit) for t in trades if t.net_credit < 0)

        # Calmar Ratio = annualized_return / max_drawdown
        # Measures return per unit of max peak-to-trough loss.
        # Higher is better; > 1.0 is generally good.
        calmar = (annualized_return * 100) / max_dd if max_dd > 0 else 0.0

        # Max consecutive losses — important for psychological/margin risk
        # Pseudocode:
        #   streak = 0; max_streak = 0
        #   for each trade: if loss → streak += 1 else streak = 0
        #   max_streak = max(max_streak, streak)
        max_consec = 0
        streak = 0
        for t in trades:
            if t.exit_pnl <= 0:
                streak += 1
                max_consec = max(max_consec, streak)
            else:
                streak = 0

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
            "sortino_ratio": sortino,
            "avg_holding_days": avg_holding,
            "total_premium_collected": total_collected,
            "total_premium_paid": total_paid,
            "calmar_ratio": calmar,
            "max_consecutive_losses": max_consec,
        }


# ── Report Generation (v2) ──────────────────────────────────────────────────

def generate_option_report(result: OptionBacktestResult) -> str:
    """Generate comprehensive text report with v2 enhancements."""
    lines = []
    sep = "=" * 70

    lines.append(sep)
    lines.append(f"  OPTION STRATEGY BACKTEST REPORT (v2)")
    lines.append(sep)
    lines.append("")

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
    lines.append(f"  Sortino Ratio:   {result.sortino_ratio:>12.2f}")
    lines.append(f"  Calmar Ratio:    {result.calmar_ratio:>12.2f}")
    lines.append(f"  Max Consec Loss: {result.max_consecutive_losses:>12d}")
    lines.append("")

    # Transaction cost analysis
    lines.append("─" * 70)
    lines.append("  TRANSACTION COSTS (Noise Filtering)")
    lines.append("─" * 70)
    lines.append(f"  Total Commission:  ${result.total_commission:>10,.2f}")
    lines.append(f"  Total Slippage:    ${result.total_slippage:>10,.2f}")
    lines.append(f"  Total Costs:       ${result.total_transaction_costs:>10,.2f}")
    cost_pct = (result.total_transaction_costs / result.initial_capital * 100
                if result.initial_capital > 0 else 0)
    lines.append(f"  Costs as % Capital:{cost_pct:>10.2f}%")
    cost_drag = (result.total_transaction_costs / abs(result.total_pnl) * 100
                 if abs(result.total_pnl) > 0 else 0)
    lines.append(f"  Cost Drag on P&L:  {cost_drag:>10.1f}%")
    lines.append("")

    # Trade stats
    lines.append("─" * 70)
    lines.append("  TRADE STATISTICS")
    lines.append("─" * 70)
    lines.append(f"  Positions Opened:{result.positions_opened:>12d}")
    lines.append(f"  Positions Closed:{result.positions_closed:>12d}")
    lines.append(f"  EV Filtered Out: {result.ev_filtered_trades:>12d}")
    lines.append(f"  Win Rate:        {result.win_rate:>12.1f}%")
    lines.append(f"  Avg Win:         ${result.avg_win:>+12,.2f}")
    lines.append(f"  Avg Loss:        ${result.avg_loss:>+12,.2f}")
    lines.append(f"  Profit Factor:   {result.profit_factor:>12.2f}")
    lines.append(f"  Avg Hold (days): {result.avg_holding_days:>12.1f}")
    lines.append("")

    # EV analysis
    lines.append("─" * 70)
    lines.append("  EXPECTED VALUE ANALYSIS")
    lines.append("─" * 70)
    lines.append(f"  Avg EV at Entry:   ${result.avg_ev_at_entry:>10,.2f}")
    lines.append(f"  Avg POP at Entry:  {result.avg_pop_at_entry:>10.1%}")
    lines.append(f"  Trades Rejected:   {result.ev_filtered_trades:>10d} (negative EV)")
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

    # Greeks alerts
    lines.append("─" * 70)
    lines.append("  GREEKS RISK MONITOR")
    lines.append("─" * 70)
    lines.append(f"  Total Greeks Alerts: {result.total_greeks_alerts}")
    if result.risk_snapshot:
        lines.append(f"  Final Portfolio:     {result.risk_snapshot}")
    lines.append("")

    # Tax compliance
    lines.append("─" * 70)
    lines.append("  TAX COMPLIANCE (Wash Sale)")
    lines.append("─" * 70)
    lines.append(f"  Wash Sale Violations:  {result.wash_sale_violations}")
    lines.append(f"  Disallowed Losses:     ${result.wash_sale_disallowed:>10,.2f}")
    lines.append("")

    # Strategy rating
    lines.append("─" * 70)
    lines.append("  STRATEGY RATING")
    lines.append("─" * 70)
    lines.append(f"  Sharpe:     {_rate_sharpe(result.sharpe_ratio)}")
    lines.append(f"  Drawdown:   {_rate_drawdown(result.max_drawdown_pct)}")
    lines.append(f"  Win Rate:   {_rate_winrate(result.win_rate)}")
    lines.append(f"  Costs:      {_rate_costs(cost_drag)}")
    lines.append(f"  Overall:    {_rate_overall(result)}")
    lines.append("")

    # Equity curve
    lines.append("─" * 70)
    lines.append("  EQUITY CURVE")
    lines.append("─" * 70)
    lines.append(_ascii_chart(result.portfolio_values, width=60, height=15))
    lines.append("")

    # Trade log
    lines.append("─" * 70)
    lines.append("  RECENT TRADES (last 20)")
    lines.append("─" * 70)
    recent = result.trades[-20:] if len(result.trades) > 20 else result.trades
    if recent:
        lines.append(f"  {'Entry':>10} → {'Exit':>10}  {'Type':<16} {'P&L':>9} {'POP':>5} {'EV':>7} {'Reason':>10}")
        lines.append("  " + "-" * 76)
        for t in recent:
            arrow = "▲" if t.exit_pnl > 0 else "▼"
            lines.append(
                f"  {t.entry_date:>10} → {t.exit_date:>10}  "
                f"{t.strategy_name:<16} "
                f"${t.exit_pnl:>+8,.0f} "
                f"{t.pop_at_entry:>4.0%} "
                f"${t.ev_at_entry:>+6,.0f} "
                f"{t.exit_reason:>10} {arrow}"
            )
    else:
        lines.append("  No completed trades.")
    lines.append("")
    lines.append(sep)

    return "\n".join(lines)


def compare_option_strategies(results: List[OptionBacktestResult]) -> str:
    """Compare multiple option strategies side by side (v2)."""
    lines = []
    sep = "=" * 90

    lines.append(sep)
    lines.append("  OPTION STRATEGY COMPARISON (v2)")
    lines.append(sep)
    lines.append("")

    header = f"  {'Metric':<25}"
    for r in results:
        header += f" {r.strategy_name:>15}"
    lines.append(header)
    lines.append("  " + "-" * (25 + 16 * len(results)))

    metrics = [
        ("Total Return %", [f"{r.total_return_pct:+.2f}%" for r in results]),
        ("Annualized Return %", [f"{r.annualized_return_pct:+.2f}%" for r in results]),
        ("Sharpe Ratio", [f"{r.sharpe_ratio:.2f}" for r in results]),
        ("Sortino Ratio", [f"{r.sortino_ratio:.2f}" for r in results]),
        ("Calmar Ratio", [f"{r.calmar_ratio:.2f}" for r in results]),
        ("Max Consec Losses", [f"{r.max_consecutive_losses}" for r in results]),
        ("Max Drawdown %", [f"{r.max_drawdown_pct:.2f}%" for r in results]),
        ("Win Rate %", [f"{r.win_rate:.1f}%" for r in results]),
        ("Profit Factor", [f"{r.profit_factor:.2f}" for r in results]),
        ("Total Trades", [f"{r.positions_closed}" for r in results]),
        ("EV Filtered Out", [f"{r.ev_filtered_trades}" for r in results]),
        ("Avg EV at Entry", [f"${r.avg_ev_at_entry:+,.0f}" for r in results]),
        ("Avg POP", [f"{r.avg_pop_at_entry:.0%}" for r in results]),
        ("Transaction Costs", [f"${r.total_transaction_costs:,.0f}" for r in results]),
        ("Greeks Alerts", [f"{r.total_greeks_alerts}" for r in results]),
        ("Wash Sales", [f"{r.wash_sale_violations}" for r in results]),
        ("Final Value $", [f"${r.final_value:,.0f}" for r in results]),
    ]

    for name, values in metrics:
        row = f"  {name:<25}"
        for v in values:
            row += f" {v:>15}"
        lines.append(row)

    lines.append("")

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


def _rate_costs(cost_drag):
    if cost_drag < 5:
        return "★★★ Minimal Cost Impact"
    elif cost_drag < 15:
        return "★★☆ Acceptable"
    elif cost_drag < 30:
        return "★☆☆ Significant"
    else:
        return "✗ Excessive — costs eroding most gains"


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
    if result.total_transaction_costs < result.initial_capital * 0.02:
        score += 1

    if score >= 6:
        return "★★★ Strong Strategy"
    elif score >= 4:
        return "★★☆ Decent Strategy"
    elif score >= 2:
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

    lines.append("  " + "            └" + "─" * len(sampled))
    return "\n".join(lines)
