"""
Tax Compliance Module: Wash Sale Detection

IRS Wash Sale Rule (Section 1091):
- If you sell a security at a loss and buy a "substantially identical"
  security within 30 days BEFORE or AFTER the sale, the loss is disallowed.
- The disallowed loss is added to the cost basis of the new purchase.

This module tracks all closed option trades and flags potential wash sales
to help traders manage tax implications.

For options specifically:
- Selling a put and being assigned is treated as a purchase
- Options on the same underlying with different strikes/expirations
  may be considered "substantially identical" (conservative interpretation)
- The IRS has not clearly defined "substantially identical" for options,
  so we use the conservative approach: same underlying = flagged

Logic (pseudocode):
─────────────────────
wash_sale_check(new_trade, trade_history):
    for each past_trade in trade_history:
        if past_trade.underlying == new_trade.underlying
           AND past_trade.pnl < 0  (closed at a loss)
           AND |new_trade.date - past_trade.close_date| <= 30 days:
            FLAG as potential wash sale
            disallowed_loss = past_trade.loss
            adjusted_basis = new_trade.entry_price + disallowed_loss
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple


@dataclass
class WashSaleAlert:
    """Alert for a potential wash sale violation."""
    new_trade_date: str
    prior_loss_date: str
    underlying: str
    prior_loss_amount: float
    days_apart: int
    disallowed_loss: float
    adjusted_basis_impact: float
    message: str

    def __repr__(self):
        return (f"WashSale({self.underlying}: ${self.disallowed_loss:+,.0f} "
                f"loss disallowed, {self.days_apart}d apart)")


@dataclass
class ClosedTrade:
    """Record of a closed trade for wash sale tracking."""
    underlying: str
    option_type: str       # "call", "put", "stock"
    strike: Optional[float]
    entry_date: str
    exit_date: str
    pnl: float
    is_loss: bool

    @property
    def exit_datetime(self) -> datetime:
        return datetime.strptime(self.exit_date, "%Y-%m-%d")


class WashSaleDetector:
    """Detects potential wash sale violations.

    Maintains a rolling window of closed trades and checks
    each new trade against the 30-day wash sale window.
    """

    def __init__(self, lookback_days: int = 30, conservative: bool = True):
        """
        Args:
            lookback_days: Wash sale window (30 days per IRS rules)
            conservative: If True, any same-underlying option counts as
                          "substantially identical". If False, only same
                          strike+type counts.
        """
        self.lookback_days = lookback_days
        self.conservative = conservative
        self.closed_trades: List[ClosedTrade] = []
        self.alerts: List[WashSaleAlert] = []
        self.total_disallowed_losses: float = 0.0

    def record_close(self, underlying: str, option_type: str,
                     strike: Optional[float], entry_date: str,
                     exit_date: str, pnl: float):
        """Record a closed trade."""
        self.closed_trades.append(ClosedTrade(
            underlying=underlying,
            option_type=option_type,
            strike=strike,
            entry_date=entry_date,
            exit_date=exit_date,
            pnl=pnl,
            is_loss=pnl < 0,
        ))

    def check_new_trade(self, underlying: str, option_type: str,
                        strike: Optional[float],
                        trade_date: str) -> List[WashSaleAlert]:
        """Check if opening a new trade triggers a wash sale.

        Looks at all losing trades closed within the wash sale window.
        """
        alerts = []
        trade_dt = datetime.strptime(trade_date, "%Y-%m-%d")

        for closed in self.closed_trades:
            if not closed.is_loss:
                continue

            # Check if same underlying
            if closed.underlying != underlying:
                continue

            # Check if substantially identical
            if not self.conservative:
                if (closed.option_type != option_type or
                        (strike is not None and closed.strike is not None and
                         abs(closed.strike - strike) > 0.01)):
                    continue

            # Check 30-day window (before AND after)
            days_apart = abs((trade_dt - closed.exit_datetime).days)
            if days_apart <= self.lookback_days:
                disallowed = abs(closed.pnl)
                alert = WashSaleAlert(
                    new_trade_date=trade_date,
                    prior_loss_date=closed.exit_date,
                    underlying=underlying,
                    prior_loss_amount=closed.pnl,
                    days_apart=days_apart,
                    disallowed_loss=disallowed,
                    adjusted_basis_impact=disallowed,
                    message=(
                        f"WASH SALE: Opening {option_type} on {underlying} "
                        f"within {days_apart} days of ${closed.pnl:+,.0f} loss "
                        f"closed on {closed.exit_date}. "
                        f"Disallowed loss: ${disallowed:,.0f} — "
                        f"this loss will be added to your new position's cost basis."
                    )
                )
                alerts.append(alert)
                self.alerts.append(alert)
                self.total_disallowed_losses += disallowed

        return alerts

    def check_trade_history(self, trades) -> List[WashSaleAlert]:
        """Analyze an entire trade history for wash sales.

        Args:
            trades: List of OptionTradeRecord from backtest results

        Returns:
            List of all wash sale alerts
        """
        self.closed_trades.clear()
        self.alerts.clear()
        self.total_disallowed_losses = 0.0

        all_alerts = []

        for trade in trades:
            # First, check if this new trade triggers a wash sale
            underlying = getattr(trade, 'strategy_name', 'unknown')
            # Use the entry as the "new trade" check point
            ws_alerts = self.check_new_trade(
                underlying="SPY",  # Simplified — all trades on same underlying
                option_type="option",
                strike=None,
                trade_date=trade.entry_date,
            )
            all_alerts.extend(ws_alerts)

            # Record this trade's close
            self.record_close(
                underlying="SPY",
                option_type="option",
                strike=None,
                entry_date=trade.entry_date,
                exit_date=trade.exit_date,
                pnl=trade.exit_pnl,
            )

        return all_alerts

    def get_summary(self) -> Dict:
        """Get wash sale summary."""
        return {
            "total_trades_analyzed": len(self.closed_trades),
            "wash_sale_violations": len(self.alerts),
            "total_disallowed_losses": self.total_disallowed_losses,
            "alerts": self.alerts,
        }


def format_wash_sale_report(detector: WashSaleDetector) -> str:
    """Generate wash sale compliance report."""
    summary = detector.get_summary()
    lines = []

    lines.append("─" * 70)
    lines.append("  TAX COMPLIANCE: WASH SALE ANALYSIS")
    lines.append("─" * 70)
    lines.append(f"  Trades Analyzed:       {summary['total_trades_analyzed']}")
    lines.append(f"  Wash Sale Violations:  {summary['wash_sale_violations']}")
    lines.append(f"  Total Disallowed Loss: ${summary['total_disallowed_losses']:>10,.2f}")
    lines.append("")

    if summary['alerts']:
        lines.append("  VIOLATIONS:")
        for i, alert in enumerate(summary['alerts'], 1):
            lines.append(f"  {i}. [{alert.new_trade_date}] {alert.message}")
        lines.append("")
        lines.append("  NOTE: Disallowed losses are added to the cost basis of the")
        lines.append("  replacement position. They are not permanently lost — they")
        lines.append("  defer the tax benefit until the replacement is sold.")
    else:
        lines.append("  No wash sale violations detected.")

    lines.append("")
    lines.append("  DISCLAIMER: This is an automated analysis for educational purposes.")
    lines.append("  Consult a tax professional for actual tax advice.")
    lines.append("")

    return "\n".join(lines)
