"""
Option Trading Strategies for QQQ/SPY

Implements the most effective real-world option strategies:
1. Covered Call / Cash-Secured Put (Wheel Strategy)
2. Iron Condor (range-bound income)
3. Vertical Spreads (directional credit/debit)
4. Straddle/Strangle (volatility plays)

Each strategy generates structured trade signals based on:
- IV analysis (sell premium when IV is high, buy when low)
- Technical signals from the underlying
- Greeks-based position management
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from options.iv_model import (
    historical_volatility, generate_iv_series, iv_rank, iv_percentile,
    classify_vol_regime, generate_iv_signals, IVSignal
)
from options.pricing import (
    call_price, put_price, delta, gamma, theta, vega, all_greeks
)


# ── Trade Signal Types ──────────────────────────────────────────────────────

@dataclass
class OptionTrade:
    """Represents a single option trade leg."""
    option_type: str     # "call" or "put"
    action: str          # "buy" or "sell"
    strike: float
    premium: float       # Per-share premium (positive = credit received)
    contracts: int = 1   # Number of contracts (1 contract = 100 shares)
    dte: int = 30        # Days to expiration at entry
    iv_at_entry: float = 0.0
    delta_at_entry: float = 0.0

    @property
    def total_premium(self) -> float:
        """Total premium for all contracts (× 100 multiplier)."""
        sign = 1 if self.action == "sell" else -1
        return sign * self.premium * self.contracts * 100

    @property
    def max_loss(self) -> float:
        """Maximum loss for this single leg (simplified)."""
        if self.action == "sell":
            if self.option_type == "call":
                return float('inf')  # Naked call = unlimited risk
            else:
                return (self.strike - self.premium) * self.contracts * 100
        else:
            return self.premium * self.contracts * 100

    def __repr__(self):
        return (f"{self.action.upper()} {self.contracts}x "
                f"{self.strike:.0f} {self.option_type.upper()} "
                f"@ {self.premium:.2f} ({self.dte}DTE)")


@dataclass
class OptionPosition:
    """A multi-leg option position."""
    strategy_name: str
    underlying_price: float
    entry_date: str
    legs: List[OptionTrade] = field(default_factory=list)
    exit_date: Optional[str] = None
    exit_pnl: float = 0.0

    @property
    def net_credit(self) -> float:
        """Net premium received (positive = credit, negative = debit)."""
        return sum(leg.total_premium for leg in self.legs)

    @property
    def max_profit(self) -> float:
        """Maximum profit potential."""
        if self.strategy_name == "iron_condor":
            return self.net_credit
        elif self.strategy_name in ("bull_put_spread", "bear_call_spread"):
            return self.net_credit
        elif self.strategy_name in ("long_straddle", "long_strangle"):
            return float('inf')
        elif self.strategy_name == "covered_call":
            return self.net_credit  # simplified
        elif self.strategy_name == "cash_secured_put":
            return self.net_credit
        return self.net_credit

    @property
    def max_loss(self) -> float:
        """Maximum loss potential."""
        if self.strategy_name == "iron_condor":
            # Width of wider spread minus net credit
            strikes = sorted([l.strike for l in self.legs])
            if len(strikes) >= 4:
                width = max(strikes[1] - strikes[0], strikes[3] - strikes[2])
                return width * 100 * self.legs[0].contracts - self.net_credit
            return 0.0
        elif self.strategy_name in ("bull_put_spread", "bear_call_spread"):
            strikes = sorted([l.strike for l in self.legs])
            if len(strikes) >= 2:
                width = strikes[-1] - strikes[0]
                return width * 100 * self.legs[0].contracts - self.net_credit
            return 0.0
        elif self.strategy_name in ("long_straddle", "long_strangle"):
            return abs(self.net_credit)  # Total debit paid
        elif self.strategy_name == "covered_call":
            return self.underlying_price * 100 - self.net_credit  # Stock can go to 0
        elif self.strategy_name == "cash_secured_put":
            put_leg = [l for l in self.legs if l.option_type == "put"][0]
            return put_leg.strike * 100 * put_leg.contracts - self.net_credit
        return 0.0

    def __repr__(self):
        legs_str = " | ".join(str(l) for l in self.legs)
        return (f"{self.strategy_name}: {legs_str} "
                f"[Net={'${:+.0f}'.format(self.net_credit)}]")


# ── Strategy Base Class ─────────────────────────────────────────────────────

class OptionStrategy(ABC):
    """Base class for option trading strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def params(self) -> Dict:
        return {}

    @abstractmethod
    def generate_trades(self, closes: List[float], highs: List[float],
                        lows: List[float], dates: List[str],
                        risk_free_rate: float = 0.04) -> List[Optional[OptionPosition]]:
        """Generate option positions for each trading day.

        Returns:
            List of OptionPosition or None for each day.
            None means no new position opened.
        """
        pass


# ── Iron Condor Strategy ────────────────────────────────────────────────────

class IronCondorStrategy(OptionStrategy):
    """Iron Condor: Sell OTM put spread + OTM call spread.

    Best when: High IV, expecting range-bound price action.
    - Sell OTM put + Buy further OTM put (bull put spread)
    - Sell OTM call + Buy further OTM call (bear call spread)
    - Profit from time decay if price stays within the short strikes.

    Entry: IV Rank > threshold
    Exit: DTE < exit_dte, or profit target hit, or max loss hit
    """

    def __init__(self, iv_rank_entry: float = 50.0, dte_target: int = 45,
                 dte_exit: int = 21, short_delta: float = 0.16,
                 wing_width: float = 5.0, profit_target: float = 0.50,
                 stop_loss: float = 2.0, contracts: int = 1):
        self.iv_rank_entry = iv_rank_entry
        self.dte_target = dte_target
        self.dte_exit = dte_exit
        self.short_delta = short_delta  # Target delta for short strikes
        self.wing_width = wing_width    # Width between short and long strikes
        self.profit_target = profit_target  # Close at 50% of max profit
        self.stop_loss = stop_loss      # Close at 2x credit received loss
        self.contracts = contracts

    @property
    def name(self) -> str:
        return "Iron Condor"

    @property
    def params(self) -> Dict:
        return {
            "iv_rank_entry": self.iv_rank_entry,
            "dte_target": self.dte_target,
            "short_delta": self.short_delta,
            "wing_width": self.wing_width,
            "profit_target": self.profit_target,
        }

    def generate_trades(self, closes, highs, lows, dates, risk_free_rate=0.04):
        n = len(closes)
        positions = [None] * n

        hv = historical_volatility(closes, window=20)
        iv_series = generate_iv_series(closes, base_iv=0.20, seed=42)

        in_position = False
        position_entry_idx = 0
        current_position = None

        for i in range(252, n):  # Need 1 year of data for IV rank
            iv_history = iv_series[max(0, i - 252):i]
            current_iv = iv_series[i]
            ivr = iv_rank(current_iv, iv_history)
            price = closes[i]
            T = self.dte_target / 365.0

            if in_position:
                # Check exit conditions
                days_held = i - position_entry_idx
                remaining_dte = self.dte_target - days_held

                if remaining_dte <= self.dte_exit:
                    # Exit at DTE target
                    current_position.exit_date = dates[i]
                    pnl = self._calc_pnl(current_position, price, remaining_dte,
                                         iv_series[i], risk_free_rate)
                    current_position.exit_pnl = pnl
                    in_position = False
                    continue

                # Check profit target / stop loss
                current_value = self._position_value(current_position, price,
                                                     remaining_dte, iv_series[i],
                                                     risk_free_rate)
                credit = current_position.net_credit
                if credit > 0:
                    profit_pct = (credit - current_value) / credit
                    if profit_pct >= self.profit_target:
                        current_position.exit_date = dates[i]
                        current_position.exit_pnl = credit - current_value
                        in_position = False
                        continue
                    if current_value > credit * self.stop_loss:
                        current_position.exit_date = dates[i]
                        current_position.exit_pnl = credit - current_value
                        in_position = False
                        continue
            else:
                # Check entry conditions
                if ivr >= self.iv_rank_entry:
                    # Find short strikes at target delta
                    put_strike = self._find_strike_by_delta(
                        price, T, risk_free_rate, current_iv,
                        target_delta=-self.short_delta, option_type="put"
                    )
                    call_strike = self._find_strike_by_delta(
                        price, T, risk_free_rate, current_iv,
                        target_delta=self.short_delta, option_type="call"
                    )

                    put_long_strike = put_strike - self.wing_width
                    call_long_strike = call_strike + self.wing_width

                    # Calculate premiums
                    short_put_prem = put_price(price, put_strike, T, risk_free_rate, current_iv)
                    long_put_prem = put_price(price, put_long_strike, T, risk_free_rate, current_iv)
                    short_call_prem = call_price(price, call_strike, T, risk_free_rate, current_iv)
                    long_call_prem = call_price(price, call_long_strike, T, risk_free_rate, current_iv)

                    position = OptionPosition(
                        strategy_name="iron_condor",
                        underlying_price=price,
                        entry_date=dates[i],
                        legs=[
                            OptionTrade("put", "sell", put_strike, short_put_prem,
                                        self.contracts, self.dte_target, current_iv,
                                        delta(price, put_strike, T, risk_free_rate, current_iv, "put")),
                            OptionTrade("put", "buy", put_long_strike, long_put_prem,
                                        self.contracts, self.dte_target, current_iv,
                                        delta(price, put_long_strike, T, risk_free_rate, current_iv, "put")),
                            OptionTrade("call", "sell", call_strike, short_call_prem,
                                        self.contracts, self.dte_target, current_iv,
                                        delta(price, call_strike, T, risk_free_rate, current_iv, "call")),
                            OptionTrade("call", "buy", call_long_strike, long_call_prem,
                                        self.contracts, self.dte_target, current_iv,
                                        delta(price, call_long_strike, T, risk_free_rate, current_iv, "call")),
                        ]
                    )

                    positions[i] = position
                    current_position = position
                    position_entry_idx = i
                    in_position = True

        return positions

    def _find_strike_by_delta(self, S, T, r, sigma, target_delta, option_type):
        """Find strike price for a given target delta."""
        best_strike = S
        best_diff = float('inf')
        # Search around the current price
        step = 1.0 if S < 200 else 2.0 if S < 500 else 5.0
        for offset in range(-40, 41):
            k = round(S / step) * step + offset * step
            if k <= 0:
                continue
            d = delta(S, k, T, r, sigma, option_type)
            diff = abs(abs(d) - abs(target_delta))
            if diff < best_diff:
                best_diff = diff
                best_strike = k
        return best_strike

    def _position_value(self, position, current_price, remaining_dte, current_iv, r):
        """Current cost to close the position."""
        T = remaining_dte / 365.0
        total = 0.0
        for leg in position.legs:
            if leg.option_type == "call":
                price = call_price(current_price, leg.strike, T, r, current_iv)
            else:
                price = put_price(current_price, leg.strike, T, r, current_iv)
            if leg.action == "sell":
                total += price * leg.contracts * 100
            else:
                total -= price * leg.contracts * 100
        return total

    def _calc_pnl(self, position, current_price, remaining_dte, current_iv, r):
        """Calculate P&L at exit."""
        close_cost = self._position_value(position, current_price, remaining_dte, current_iv, r)
        return position.net_credit - close_cost


# ── Vertical Spread Strategy ────────────────────────────────────────────────

class VerticalSpreadStrategy(OptionStrategy):
    """Bull Put Spread or Bear Call Spread based on trend + IV.

    Combines directional bias with premium selling:
    - Uptrend + High IV → Bull Put Spread (sell put spread below market)
    - Downtrend + High IV → Bear Call Spread (sell call spread above market)
    """

    def __init__(self, ma_fast: int = 20, ma_slow: int = 50,
                 iv_rank_entry: float = 40.0, dte_target: int = 30,
                 dte_exit: int = 10, spread_width: float = 5.0,
                 short_delta: float = 0.30, profit_target: float = 0.65,
                 contracts: int = 1):
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.iv_rank_entry = iv_rank_entry
        self.dte_target = dte_target
        self.dte_exit = dte_exit
        self.spread_width = spread_width
        self.short_delta = short_delta
        self.profit_target = profit_target
        self.contracts = contracts

    @property
    def name(self) -> str:
        return "Vertical Spread"

    @property
    def params(self) -> Dict:
        return {
            "ma_fast": self.ma_fast,
            "ma_slow": self.ma_slow,
            "iv_rank_entry": self.iv_rank_entry,
            "spread_width": self.spread_width,
            "short_delta": self.short_delta,
        }

    def _sma(self, values, window):
        result = [0.0] * len(values)
        for i in range(window - 1, len(values)):
            result[i] = sum(values[i - window + 1:i + 1]) / window
        return result

    def generate_trades(self, closes, highs, lows, dates, risk_free_rate=0.04):
        n = len(closes)
        positions = [None] * n

        hv = historical_volatility(closes, window=20)
        iv_series = generate_iv_series(closes, base_iv=0.20, seed=42)
        fast_ma = self._sma(closes, self.ma_fast)
        slow_ma = self._sma(closes, self.ma_slow)

        in_position = False
        position_entry_idx = 0
        current_position = None

        for i in range(252, n):
            iv_history = iv_series[max(0, i - 252):i]
            current_iv = iv_series[i]
            ivr = iv_rank(current_iv, iv_history)
            price = closes[i]
            T = self.dte_target / 365.0

            if in_position:
                days_held = i - position_entry_idx
                remaining_dte = self.dte_target - days_held

                if remaining_dte <= self.dte_exit:
                    current_position.exit_date = dates[i]
                    pnl = self._calc_pnl(current_position, price, remaining_dte,
                                         iv_series[i], risk_free_rate)
                    current_position.exit_pnl = pnl
                    in_position = False
                    continue

                # Profit target check
                current_value = self._position_value(current_position, price,
                                                     remaining_dte, iv_series[i],
                                                     risk_free_rate)
                credit = current_position.net_credit
                if credit > 0:
                    profit_pct = (credit - current_value) / credit
                    if profit_pct >= self.profit_target:
                        current_position.exit_date = dates[i]
                        current_position.exit_pnl = credit - current_value
                        in_position = False
                        continue
            else:
                if ivr >= self.iv_rank_entry and fast_ma[i] > 0 and slow_ma[i] > 0:
                    is_uptrend = fast_ma[i] > slow_ma[i]

                    if is_uptrend:
                        # Bull Put Spread
                        short_strike = self._find_strike_by_delta(
                            price, T, risk_free_rate, current_iv,
                            target_delta=self.short_delta, option_type="put"
                        )
                        long_strike = short_strike - self.spread_width

                        short_prem = put_price(price, short_strike, T, risk_free_rate, current_iv)
                        long_prem = put_price(price, long_strike, T, risk_free_rate, current_iv)

                        position = OptionPosition(
                            strategy_name="bull_put_spread",
                            underlying_price=price,
                            entry_date=dates[i],
                            legs=[
                                OptionTrade("put", "sell", short_strike, short_prem,
                                            self.contracts, self.dte_target, current_iv,
                                            delta(price, short_strike, T, risk_free_rate, current_iv, "put")),
                                OptionTrade("put", "buy", long_strike, long_prem,
                                            self.contracts, self.dte_target, current_iv,
                                            delta(price, long_strike, T, risk_free_rate, current_iv, "put")),
                            ]
                        )
                    else:
                        # Bear Call Spread
                        short_strike = self._find_strike_by_delta(
                            price, T, risk_free_rate, current_iv,
                            target_delta=self.short_delta, option_type="call"
                        )
                        long_strike = short_strike + self.spread_width

                        short_prem = call_price(price, short_strike, T, risk_free_rate, current_iv)
                        long_prem = call_price(price, long_strike, T, risk_free_rate, current_iv)

                        position = OptionPosition(
                            strategy_name="bear_call_spread",
                            underlying_price=price,
                            entry_date=dates[i],
                            legs=[
                                OptionTrade("call", "sell", short_strike, short_prem,
                                            self.contracts, self.dte_target, current_iv,
                                            delta(price, short_strike, T, risk_free_rate, current_iv, "call")),
                                OptionTrade("call", "buy", long_strike, long_prem,
                                            self.contracts, self.dte_target, current_iv,
                                            delta(price, long_strike, T, risk_free_rate, current_iv, "call")),
                            ]
                        )

                    positions[i] = position
                    current_position = position
                    position_entry_idx = i
                    in_position = True

        return positions

    def _find_strike_by_delta(self, S, T, r, sigma, target_delta, option_type):
        best_strike = S
        best_diff = float('inf')
        step = 1.0 if S < 200 else 2.0 if S < 500 else 5.0
        for offset in range(-40, 41):
            k = round(S / step) * step + offset * step
            if k <= 0:
                continue
            d = delta(S, k, T, r, sigma, option_type)
            diff = abs(abs(d) - abs(target_delta))
            if diff < best_diff:
                best_diff = diff
                best_strike = k
        return best_strike

    def _position_value(self, position, current_price, remaining_dte, current_iv, r):
        T = remaining_dte / 365.0
        total = 0.0
        for leg in position.legs:
            if leg.option_type == "call":
                price = call_price(current_price, leg.strike, T, r, current_iv)
            else:
                price = put_price(current_price, leg.strike, T, r, current_iv)
            if leg.action == "sell":
                total += price * leg.contracts * 100
            else:
                total -= price * leg.contracts * 100
        return total

    def _calc_pnl(self, position, current_price, remaining_dte, current_iv, r):
        close_cost = self._position_value(position, current_price, remaining_dte, current_iv, r)
        return position.net_credit - close_cost


# ── Wheel Strategy ──────────────────────────────────────────────────────────

class WheelStrategy(OptionStrategy):
    """The Wheel: Sell CSPs → Get assigned → Sell Covered Calls → Repeat.

    One of the most popular income strategies for SPY/QQQ:
    Phase 1: Sell cash-secured puts at support levels
    Phase 2: If assigned, sell covered calls above cost basis
    Phase 3: If called away, go back to Phase 1

    Entry: IV Rank > threshold, underlying near support
    """

    def __init__(self, iv_rank_entry: float = 30.0, dte_target: int = 30,
                 put_delta: float = 0.30, call_delta: float = 0.30,
                 contracts: int = 1):
        self.iv_rank_entry = iv_rank_entry
        self.dte_target = dte_target
        self.put_delta = put_delta
        self.call_delta = call_delta
        self.contracts = contracts

    @property
    def name(self) -> str:
        return "Wheel Strategy"

    @property
    def params(self) -> Dict:
        return {
            "iv_rank_entry": self.iv_rank_entry,
            "dte_target": self.dte_target,
            "put_delta": self.put_delta,
            "call_delta": self.call_delta,
        }

    def generate_trades(self, closes, highs, lows, dates, risk_free_rate=0.04):
        n = len(closes)
        positions = [None] * n

        hv = historical_volatility(closes, window=20)
        iv_series = generate_iv_series(closes, base_iv=0.20, seed=42)

        # State machine: "cash" → "csp_open" → "assigned" → "cc_open" → "cash"
        state = "cash"
        position_entry_idx = 0
        current_position = None
        cost_basis = 0.0  # Track cost basis when assigned

        for i in range(252, n):
            iv_history = iv_series[max(0, i - 252):i]
            current_iv = iv_series[i]
            ivr = iv_rank(current_iv, iv_history)
            price = closes[i]
            T = self.dte_target / 365.0

            if state == "cash":
                # Phase 1: Look to sell CSP
                if ivr >= self.iv_rank_entry:
                    put_strike = self._find_strike_by_delta(
                        price, T, risk_free_rate, current_iv,
                        target_delta=self.put_delta, option_type="put"
                    )
                    premium = put_price(price, put_strike, T, risk_free_rate, current_iv)

                    position = OptionPosition(
                        strategy_name="cash_secured_put",
                        underlying_price=price,
                        entry_date=dates[i],
                        legs=[
                            OptionTrade("put", "sell", put_strike, premium,
                                        self.contracts, self.dte_target, current_iv,
                                        delta(price, put_strike, T, risk_free_rate, current_iv, "put")),
                        ]
                    )
                    positions[i] = position
                    current_position = position
                    position_entry_idx = i
                    state = "csp_open"

            elif state == "csp_open":
                days_held = i - position_entry_idx
                remaining_dte = self.dte_target - days_held
                put_leg = current_position.legs[0]

                if remaining_dte <= 0:
                    # Expiration day
                    if price < put_leg.strike:
                        # Assigned! Now holding stock
                        cost_basis = put_leg.strike - put_leg.premium
                        current_position.exit_date = dates[i]
                        current_position.exit_pnl = -(put_leg.strike - price - put_leg.premium) * 100 * self.contracts
                        state = "assigned"
                    else:
                        # Expired worthless - full profit
                        current_position.exit_date = dates[i]
                        current_position.exit_pnl = put_leg.premium * 100 * self.contracts
                        state = "cash"

            elif state == "assigned":
                # Phase 2: Sell covered call above cost basis
                call_strike = self._find_strike_by_delta(
                    price, T, risk_free_rate, current_iv,
                    target_delta=self.call_delta, option_type="call"
                )
                # Ensure strike is above cost basis
                call_strike = max(call_strike, math.ceil(cost_basis))

                premium = call_price(price, call_strike, T, risk_free_rate, current_iv)

                position = OptionPosition(
                    strategy_name="covered_call",
                    underlying_price=price,
                    entry_date=dates[i],
                    legs=[
                        OptionTrade("call", "sell", call_strike, premium,
                                    self.contracts, self.dte_target, current_iv,
                                    delta(price, call_strike, T, risk_free_rate, current_iv, "call")),
                    ]
                )
                positions[i] = position
                current_position = position
                position_entry_idx = i
                state = "cc_open"

            elif state == "cc_open":
                days_held = i - position_entry_idx
                remaining_dte = self.dte_target - days_held
                call_leg = current_position.legs[0]

                if remaining_dte <= 0:
                    if price > call_leg.strike:
                        # Called away - stock sold at strike + premium
                        pnl = (call_leg.strike - cost_basis + call_leg.premium) * 100 * self.contracts
                        current_position.exit_date = dates[i]
                        current_position.exit_pnl = pnl
                        state = "cash"
                    else:
                        # Expired worthless - keep premium, still hold stock
                        current_position.exit_date = dates[i]
                        current_position.exit_pnl = call_leg.premium * 100 * self.contracts
                        cost_basis -= call_leg.premium  # Lower cost basis
                        state = "assigned"

        return positions

    def _find_strike_by_delta(self, S, T, r, sigma, target_delta, option_type):
        best_strike = S
        best_diff = float('inf')
        step = 1.0 if S < 200 else 2.0 if S < 500 else 5.0
        for offset in range(-40, 41):
            k = round(S / step) * step + offset * step
            if k <= 0:
                continue
            d = delta(S, k, T, r, sigma, option_type)
            diff = abs(abs(d) - abs(target_delta))
            if diff < best_diff:
                best_diff = diff
                best_strike = k
        return best_strike


# ── Straddle/Strangle Strategy ──────────────────────────────────────────────

class StraddleStrategy(OptionStrategy):
    """Long Straddle/Strangle: Buy ATM/OTM puts + calls for volatility expansion.

    Best when: Low IV, expecting a big move (either direction).
    - Straddle: Buy ATM call + ATM put (same strike)
    - Strangle: Buy OTM call + OTM put (different strikes, cheaper)

    Entry: IV Rank < threshold (cheap premium)
    Exit: Profit target on vol expansion, or DTE exit, or stop loss
    """

    def __init__(self, iv_rank_entry: float = 30.0, dte_target: int = 45,
                 dte_exit: int = 14, use_strangle: bool = True,
                 strangle_width: float = 5.0, profit_target: float = 1.0,
                 stop_loss: float = 0.50, contracts: int = 1):
        self.iv_rank_entry = iv_rank_entry  # Enter when IV rank BELOW this
        self.dte_target = dte_target
        self.dte_exit = dte_exit
        self.use_strangle = use_strangle
        self.strangle_width = strangle_width
        self.profit_target = profit_target  # 100% profit target
        self.stop_loss = stop_loss          # Close if lost 50% of debit
        self.contracts = contracts

    @property
    def name(self) -> str:
        return "Long Straddle" if not self.use_strangle else "Long Strangle"

    @property
    def params(self) -> Dict:
        return {
            "iv_rank_entry": self.iv_rank_entry,
            "dte_target": self.dte_target,
            "use_strangle": self.use_strangle,
            "profit_target": self.profit_target,
        }

    def generate_trades(self, closes, highs, lows, dates, risk_free_rate=0.04):
        n = len(closes)
        positions = [None] * n

        iv_series = generate_iv_series(closes, base_iv=0.20, seed=42)

        in_position = False
        position_entry_idx = 0
        current_position = None

        for i in range(252, n):
            iv_history = iv_series[max(0, i - 252):i]
            current_iv = iv_series[i]
            ivr = iv_rank(current_iv, iv_history)
            price = closes[i]
            T = self.dte_target / 365.0

            if in_position:
                days_held = i - position_entry_idx
                remaining_dte = self.dte_target - days_held

                if remaining_dte <= self.dte_exit:
                    current_position.exit_date = dates[i]
                    pnl = self._calc_pnl(current_position, price, remaining_dte,
                                         iv_series[i], risk_free_rate)
                    current_position.exit_pnl = pnl
                    in_position = False
                    continue

                # Check profit/loss targets
                current_value = self._position_value(current_position, price,
                                                     remaining_dte, iv_series[i],
                                                     risk_free_rate)
                debit_paid = abs(current_position.net_credit)
                if debit_paid > 0:
                    profit = current_value - debit_paid
                    if profit >= debit_paid * self.profit_target:
                        current_position.exit_date = dates[i]
                        current_position.exit_pnl = profit
                        in_position = False
                        continue
                    if profit <= -debit_paid * self.stop_loss:
                        current_position.exit_date = dates[i]
                        current_position.exit_pnl = profit
                        in_position = False
                        continue
            else:
                # Enter when IV is LOW (cheap premium)
                if ivr <= self.iv_rank_entry:
                    step = 1.0 if price < 200 else 2.0 if price < 500 else 5.0
                    atm_strike = round(price / step) * step

                    if self.use_strangle:
                        put_strike = atm_strike - self.strangle_width
                        call_strike = atm_strike + self.strangle_width
                    else:
                        put_strike = atm_strike
                        call_strike = atm_strike

                    call_prem = call_price(price, call_strike, T, risk_free_rate, current_iv)
                    put_prem = put_price(price, put_strike, T, risk_free_rate, current_iv)

                    position = OptionPosition(
                        strategy_name="long_strangle" if self.use_strangle else "long_straddle",
                        underlying_price=price,
                        entry_date=dates[i],
                        legs=[
                            OptionTrade("call", "buy", call_strike, call_prem,
                                        self.contracts, self.dte_target, current_iv,
                                        delta(price, call_strike, T, risk_free_rate, current_iv, "call")),
                            OptionTrade("put", "buy", put_strike, put_prem,
                                        self.contracts, self.dte_target, current_iv,
                                        delta(price, put_strike, T, risk_free_rate, current_iv, "put")),
                        ]
                    )

                    positions[i] = position
                    current_position = position
                    position_entry_idx = i
                    in_position = True

        return positions

    def _position_value(self, position, current_price, remaining_dte, current_iv, r):
        T = remaining_dte / 365.0
        total = 0.0
        for leg in position.legs:
            if leg.option_type == "call":
                price = call_price(current_price, leg.strike, T, r, current_iv)
            else:
                price = put_price(current_price, leg.strike, T, r, current_iv)
            total += price * leg.contracts * 100  # All legs are long
        return total

    def _calc_pnl(self, position, current_price, remaining_dte, current_iv, r):
        current_value = self._position_value(position, current_price, remaining_dte, current_iv, r)
        debit_paid = abs(position.net_credit)
        return current_value - debit_paid


# ── IV-Adaptive Strategy ────────────────────────────────────────────────────

class IVAdaptiveStrategy(OptionStrategy):
    """Adaptive strategy that switches between selling and buying premium
    based on IV regime.

    This is the "smart" meta-strategy:
    - High IV → Iron Condor (sell premium)
    - Normal IV → Vertical Spreads (directional + premium)
    - Low IV → Long Straddle (buy premium)

    The IV-Adaptive approach is considered one of the best systematic
    option strategies because it adapts to market conditions.
    """

    def __init__(self, high_iv_threshold: float = 60.0,
                 low_iv_threshold: float = 25.0,
                 dte_target: int = 30, contracts: int = 1):
        self.high_iv_threshold = high_iv_threshold
        self.low_iv_threshold = low_iv_threshold
        self.dte_target = dte_target
        self.contracts = contracts

        # Sub-strategies
        self._ic = IronCondorStrategy(
            iv_rank_entry=high_iv_threshold, dte_target=dte_target,
            contracts=contracts
        )
        self._vs = VerticalSpreadStrategy(
            iv_rank_entry=30.0, dte_target=dte_target,
            contracts=contracts
        )
        self._straddle = StraddleStrategy(
            iv_rank_entry=low_iv_threshold, dte_target=dte_target + 15,
            contracts=contracts
        )

    @property
    def name(self) -> str:
        return "IV-Adaptive"

    @property
    def params(self) -> Dict:
        return {
            "high_iv_threshold": self.high_iv_threshold,
            "low_iv_threshold": self.low_iv_threshold,
            "dte_target": self.dte_target,
        }

    def generate_trades(self, closes, highs, lows, dates, risk_free_rate=0.04):
        """Delegate to sub-strategies based on IV regime."""
        n = len(closes)

        iv_series = generate_iv_series(closes, base_iv=0.20, seed=42)

        # Generate trades from all sub-strategies
        ic_trades = self._ic.generate_trades(closes, highs, lows, dates, risk_free_rate)
        vs_trades = self._vs.generate_trades(closes, highs, lows, dates, risk_free_rate)
        st_trades = self._straddle.generate_trades(closes, highs, lows, dates, risk_free_rate)

        # Merge: pick the one that matches current regime
        positions = [None] * n
        in_position = False
        position_exit_day = 0

        for i in range(252, n):
            if in_position and i < position_exit_day:
                continue
            in_position = False

            iv_history = iv_series[max(0, i - 252):i]
            current_iv = iv_series[i]
            ivr = iv_rank(current_iv, iv_history)

            # Pick strategy based on IV regime
            selected = None
            if ivr >= self.high_iv_threshold and ic_trades[i] is not None:
                selected = ic_trades[i]
            elif ivr <= self.low_iv_threshold and st_trades[i] is not None:
                selected = st_trades[i]
            elif vs_trades[i] is not None:
                selected = vs_trades[i]

            if selected is not None:
                positions[i] = selected
                in_position = True
                position_exit_day = i + self.dte_target

        return positions
