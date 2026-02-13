"""
Expected Value (EV) Framework for Option Trade Selection

Based on principles from Euan Sinclair's "Option Trading":
- Only enter trades with positive expected value
- EV = P(win) × avg_win - P(loss) × avg_loss
- Use probability of profit (POP) from Black-Scholes delta
- Adjust for realistic transaction costs and slippage

Mathematical Framework (pseudocode):
─────────────────────────────────────
1. Probability of Profit (POP):
   For credit strategies (sell premium):
     POP_call = 1 - N(d2)     [prob that call expires OTM]
     POP_put  = N(-d2)        [prob that put expires OTM]
     where d2 = (ln(S/K) + (r - σ²/2)T) / (σ√T)

   For Iron Condor:
     POP ≈ POP_put_spread × POP_call_spread

2. Expected Value:
   EV = POP × max_profit - (1 - POP) × max_loss
   EV_adjusted = EV - transaction_costs

3. Edge (Sinclair's approach):
   Theoretical price = BS_price(realized_vol)
   Market price = BS_price(implied_vol)
   Edge = |market_price - theoretical_price|
   If selling: edge = market_price - theoretical_price (want overpriced)
   If buying: edge = theoretical_price - market_price (want underpriced)

4. Kelly Criterion (optimal position size):
   f* = edge / max_loss
   Use fractional Kelly (25-50%) for safety
"""

import math
from dataclasses import dataclass
from typing import Optional, List, Dict

from options.pricing import (
    call_price, put_price, delta, _norm_cdf, _d1_d2, option_price
)


@dataclass
class EVResult:
    """Expected value analysis for a potential trade."""
    strategy_name: str
    pop: float              # Probability of profit (0-1)
    max_profit: float       # Maximum profit ($)
    max_loss: float         # Maximum loss ($)
    expected_value: float   # Raw EV ($)
    ev_after_costs: float   # EV after slippage + commission ($)
    edge_pct: float         # Edge as % of capital at risk
    kelly_fraction: float   # Kelly criterion optimal sizing
    kelly_half: float       # Half-Kelly (conservative)
    is_positive_ev: bool    # Whether trade has +EV
    breakeven_prices: List[float]  # Breakeven underlying prices
    explanation: str        # Human-readable explanation

    def __repr__(self):
        ev_sign = "+" if self.is_positive_ev else "-"
        return (f"EV({self.strategy_name}: POP={self.pop:.1%}, "
                f"EV=${self.expected_value:+,.0f}, "
                f"Edge={self.edge_pct:+.2f}%, "
                f"{'PASS' if self.is_positive_ev else 'REJECT'})")


class EVAnalyzer:
    """Expected Value analyzer for option trades.

    Implements Euan Sinclair's edge-finding framework:
    1. Calculate theoretical value using realized (historical) volatility
    2. Compare to market price (implied volatility)
    3. Only trade when there's a quantifiable edge
    """

    def __init__(self, commission_per_contract: float = 0.65,
                 slippage_pct: float = 0.10,
                 min_ev_threshold: float = 0.0,
                 min_pop: float = 0.0,
                 risk_free_rate: float = 0.04):
        """
        Args:
            commission_per_contract: Per contract per leg ($)
            slippage_pct: Slippage as fraction of bid-ask spread
            min_ev_threshold: Minimum EV to accept trade ($)
            min_pop: Minimum probability of profit (0-1)
            risk_free_rate: Risk-free rate
        """
        self.commission_per_contract = commission_per_contract
        self.slippage_pct = slippage_pct
        self.min_ev_threshold = min_ev_threshold
        self.min_pop = min_pop
        self.risk_free_rate = risk_free_rate

    def probability_of_profit_credit_spread(
        self, S: float, K_short: float, K_long: float,
        T: float, sigma: float, net_credit: float,
        spread_type: str = "put"
    ) -> float:
        """POP for a credit spread.

        Pseudocode:
            For bull put spread (credit):
              breakeven = K_short - net_credit_per_share
              POP = P(S > breakeven at expiry)
                  = N(d2) where K = breakeven

            For bear call spread (credit):
              breakeven = K_short + net_credit_per_share
              POP = P(S < breakeven at expiry)
                  = N(-d2) where K = breakeven
        """
        r = self.risk_free_rate

        if spread_type == "put":
            breakeven = K_short - net_credit / 100.0
            d1, d2 = _d1_d2(S, breakeven, T, r, sigma)
            pop = _norm_cdf(d2)  # P(S > breakeven)
        else:  # call
            breakeven = K_short + net_credit / 100.0
            d1, d2 = _d1_d2(S, breakeven, T, r, sigma)
            pop = _norm_cdf(-d2)  # P(S < breakeven)

        return max(0.0, min(1.0, pop))

    def probability_of_profit_iron_condor(
        self, S: float, K_put_short: float, K_put_long: float,
        K_call_short: float, K_call_long: float,
        T: float, sigma: float, net_credit: float
    ) -> float:
        """POP for an iron condor.

        Pseudocode:
            lower_breakeven = K_put_short - net_credit_per_share
            upper_breakeven = K_call_short + net_credit_per_share
            POP = P(lower_BE < S < upper_BE at expiry)
                = N(d2_upper) - N(d2_lower)
        """
        r = self.risk_free_rate
        credit_per_share = net_credit / 100.0

        lower_be = K_put_short - credit_per_share
        upper_be = K_call_short + credit_per_share

        _, d2_upper = _d1_d2(S, upper_be, T, r, sigma)
        _, d2_lower = _d1_d2(S, lower_be, T, r, sigma)

        pop = _norm_cdf(d2_upper) - _norm_cdf(d2_lower)
        return max(0.0, min(1.0, pop))

    def probability_of_profit_straddle(
        self, S: float, K: float, T: float, sigma: float,
        total_debit: float
    ) -> float:
        """POP for a long straddle/strangle.

        Pseudocode:
            debit_per_share = total_debit / 100
            upper_breakeven = K + debit_per_share (for straddle)
            lower_breakeven = K - debit_per_share
            POP = P(S > upper_BE) + P(S < lower_BE)
                = (1 - N(d2_upper)) + N(d2_lower) where d2 uses breakeven as K
        """
        r = self.risk_free_rate
        debit_per_share = total_debit / 100.0

        upper_be = K + debit_per_share
        lower_be = K - debit_per_share

        _, d2_upper = _d1_d2(S, upper_be, T, r, sigma)
        _, d2_lower = _d1_d2(S, lower_be, T, r, sigma)

        pop = (1.0 - _norm_cdf(d2_upper)) + _norm_cdf(-d2_lower)
        return max(0.0, min(1.0, pop))

    def calculate_edge(self, S: float, K: float, T: float,
                       implied_vol: float, realized_vol: float,
                       option_type: str = "call") -> float:
        """Calculate the volatility edge (Sinclair's approach).

        Pseudocode:
            theoretical_price = BS(S, K, T, r, realized_vol)
            market_price = BS(S, K, T, r, implied_vol)
            edge = market_price - theoretical_price
            If selling options: positive edge means overpriced (good to sell)
            If buying options: negative edge means underpriced (good to buy)
        """
        r = self.risk_free_rate
        market = option_price(S, K, T, r, implied_vol, option_type)
        theoretical = option_price(S, K, T, r, realized_vol, option_type)
        return market - theoretical

    def analyze_iron_condor(
        self, S: float, K_put_short: float, K_put_long: float,
        K_call_short: float, K_call_long: float,
        T: float, iv: float, hv: float, contracts: int = 1
    ) -> EVResult:
        """Full EV analysis for an Iron Condor."""
        r = self.risk_free_rate

        # Calculate premiums
        sp = put_price(S, K_put_short, T, r, iv)
        lp = put_price(S, K_put_long, T, r, iv)
        sc = call_price(S, K_call_short, T, r, iv)
        lc = call_price(S, K_call_long, T, r, iv)

        net_credit = (sp - lp + sc - lc) * contracts * 100
        put_width = (K_put_short - K_put_long) * contracts * 100
        call_width = (K_call_long - K_call_short) * contracts * 100
        max_loss = max(put_width, call_width) - net_credit
        max_profit = net_credit

        # Transaction costs
        num_legs = 4
        total_commission = self.commission_per_contract * contracts * num_legs * 2  # open + close
        slippage = net_credit * self.slippage_pct

        # POP
        pop = self.probability_of_profit_iron_condor(
            S, K_put_short, K_put_long, K_call_short, K_call_long,
            T, iv, net_credit
        )

        # EV
        ev_raw = pop * max_profit - (1 - pop) * max_loss
        ev_adjusted = ev_raw - total_commission - slippage

        # Edge
        edge_pct = (ev_adjusted / max_loss * 100) if max_loss > 0 else 0

        # Kelly criterion
        if max_loss > 0 and pop > 0:
            b = max_profit / max_loss  # odds
            kelly = (pop * (b + 1) - 1) / b if b > 0 else 0
        else:
            kelly = 0
        kelly = max(0.0, min(kelly, 1.0))

        breakevens = [K_put_short - net_credit / (contracts * 100),
                      K_call_short + net_credit / (contracts * 100)]

        explanation = (
            f"Iron Condor: Sell {K_put_short:.0f}/{K_put_long:.0f}P "
            f"+ {K_call_short:.0f}/{K_call_long:.0f}C\n"
            f"  Credit: ${net_credit:,.0f} | Max Loss: ${max_loss:,.0f}\n"
            f"  POP: {pop:.1%} | EV: ${ev_adjusted:+,.0f}\n"
            f"  IV={iv:.1%} vs HV={hv:.1%} → "
            f"{'Premium is RICH (sell)' if iv > hv else 'Premium is CHEAP (avoid selling)'}\n"
            f"  Breakevens: ${breakevens[0]:.2f} - ${breakevens[1]:.2f}\n"
            f"  Costs: commission ${total_commission:.0f} + slippage ${slippage:.0f}"
        )

        return EVResult(
            strategy_name="iron_condor",
            pop=pop,
            max_profit=max_profit,
            max_loss=max_loss,
            expected_value=ev_raw,
            ev_after_costs=ev_adjusted,
            edge_pct=edge_pct,
            kelly_fraction=kelly,
            kelly_half=kelly * 0.5,
            is_positive_ev=ev_adjusted > self.min_ev_threshold and pop > self.min_pop,
            breakeven_prices=breakevens,
            explanation=explanation,
        )

    def analyze_credit_spread(
        self, S: float, K_short: float, K_long: float,
        T: float, iv: float, hv: float,
        spread_type: str = "put", contracts: int = 1
    ) -> EVResult:
        """Full EV analysis for a vertical credit spread."""
        r = self.risk_free_rate

        if spread_type == "put":
            short_prem = put_price(S, K_short, T, r, iv)
            long_prem = put_price(S, K_long, T, r, iv)
            width = K_short - K_long
            name = "bull_put_spread"
        else:
            short_prem = call_price(S, K_short, T, r, iv)
            long_prem = call_price(S, K_long, T, r, iv)
            width = K_long - K_short
            name = "bear_call_spread"

        net_credit = (short_prem - long_prem) * contracts * 100
        max_loss = width * contracts * 100 - net_credit
        max_profit = net_credit

        total_commission = self.commission_per_contract * contracts * 2 * 2
        slippage = net_credit * self.slippage_pct

        pop = self.probability_of_profit_credit_spread(
            S, K_short, K_long, T, iv, net_credit, spread_type
        )

        ev_raw = pop * max_profit - (1 - pop) * max_loss
        ev_adjusted = ev_raw - total_commission - slippage

        edge_pct = (ev_adjusted / max_loss * 100) if max_loss > 0 else 0

        if max_loss > 0 and pop > 0:
            b = max_profit / max_loss
            kelly = (pop * (b + 1) - 1) / b if b > 0 else 0
        else:
            kelly = 0
        kelly = max(0.0, min(kelly, 1.0))

        if spread_type == "put":
            breakevens = [K_short - net_credit / (contracts * 100)]
        else:
            breakevens = [K_short + net_credit / (contracts * 100)]

        explanation = (
            f"{'Bull Put' if spread_type == 'put' else 'Bear Call'} Spread: "
            f"Sell {K_short:.0f} / Buy {K_long:.0f}\n"
            f"  Credit: ${net_credit:,.0f} | Max Loss: ${max_loss:,.0f}\n"
            f"  POP: {pop:.1%} | EV: ${ev_adjusted:+,.0f}\n"
            f"  IV={iv:.1%} vs HV={hv:.1%}\n"
            f"  Breakeven: ${breakevens[0]:.2f}"
        )

        return EVResult(
            strategy_name=name,
            pop=pop,
            max_profit=max_profit,
            max_loss=max_loss,
            expected_value=ev_raw,
            ev_after_costs=ev_adjusted,
            edge_pct=edge_pct,
            kelly_fraction=kelly,
            kelly_half=kelly * 0.5,
            is_positive_ev=ev_adjusted > self.min_ev_threshold and pop > self.min_pop,
            breakeven_prices=breakevens,
            explanation=explanation,
        )

    def analyze_long_straddle(
        self, S: float, K_call: float, K_put: float,
        T: float, iv: float, hv: float, contracts: int = 1
    ) -> EVResult:
        """Full EV analysis for a long straddle/strangle."""
        r = self.risk_free_rate

        call_prem = call_price(S, K_call, T, r, iv)
        put_prem = put_price(S, K_put, T, r, iv)
        total_debit = (call_prem + put_prem) * contracts * 100
        max_loss = total_debit

        total_commission = self.commission_per_contract * contracts * 2 * 2
        slippage = total_debit * self.slippage_pct

        K_mid = (K_call + K_put) / 2
        pop = self.probability_of_profit_straddle(S, K_mid, T, iv, total_debit)

        # For straddle, max profit is theoretically unlimited
        # Use 2× debit as expected win for EV calculation
        expected_win = total_debit * 2.0
        ev_raw = pop * expected_win - (1 - pop) * max_loss
        ev_adjusted = ev_raw - total_commission - slippage

        edge_pct = (ev_adjusted / max_loss * 100) if max_loss > 0 else 0

        kelly = max(0.0, ev_adjusted / max_loss) if max_loss > 0 else 0
        kelly = min(kelly, 1.0)

        debit_per_share = total_debit / (contracts * 100)
        breakevens = [K_put - debit_per_share, K_call + debit_per_share]

        is_straddle = abs(K_call - K_put) < 0.01
        name = "long_straddle" if is_straddle else "long_strangle"

        explanation = (
            f"{'Straddle' if is_straddle else 'Strangle'}: "
            f"Buy {K_call:.0f}C + {K_put:.0f}P\n"
            f"  Debit: ${total_debit:,.0f} | Max Loss: ${max_loss:,.0f}\n"
            f"  POP: {pop:.1%} | EV: ${ev_adjusted:+,.0f}\n"
            f"  IV={iv:.1%} vs HV={hv:.1%} → "
            f"{'Premium is CHEAP (good for buying)' if iv < hv else 'Premium is EXPENSIVE (risky)'}\n"
            f"  Breakevens: ${breakevens[0]:.2f} / ${breakevens[1]:.2f}"
        )

        return EVResult(
            strategy_name=name,
            pop=pop,
            max_profit=float('inf'),
            max_loss=max_loss,
            expected_value=ev_raw,
            ev_after_costs=ev_adjusted,
            edge_pct=edge_pct,
            kelly_fraction=kelly,
            kelly_half=kelly * 0.5,
            is_positive_ev=ev_adjusted > self.min_ev_threshold and pop > self.min_pop,
            breakeven_prices=breakevens,
            explanation=explanation,
        )


def format_ev_report(ev: EVResult) -> str:
    """Format EV analysis into a readable report."""
    lines = []
    lines.append("─" * 60)
    lines.append("  EXPECTED VALUE ANALYSIS")
    lines.append("─" * 60)
    lines.append(f"  {ev.explanation}")
    lines.append("")
    lines.append(f"  Verdict: {'POSITIVE EV - TRADE' if ev.is_positive_ev else 'NEGATIVE EV - SKIP'}")
    if ev.kelly_half > 0:
        lines.append(f"  Position Size (½ Kelly): {ev.kelly_half:.1%} of capital")
    lines.append("─" * 60)
    return "\n".join(lines)
