"""
Option Trading Strategies for QQQ/SPY

Implements the most effective real-world option strategies:
1. Covered Call / Cash-Secured Put (Wheel Strategy)
2. Iron Condor (range-bound income)
3. Vertical Spreads (directional credit/debit)
4. Straddle/Strangle (volatility plays)
5. Butterfly Spread (pinning plays)
6. PCR-Enhanced Iron Condor (sentiment-filtered)
7. IV-Adaptive (meta-strategy)
8. TA-Driven (technical-analysis entries)

Each strategy generates structured trade signals based on:
- IV analysis (sell premium when IV is high, buy when low)
- Technical signals from the underlying
- Greeks-based position management

Architecture notes:
- Every strategy accepts `ivr_lookback` (default 60) and `max_concurrent`
  (default 3) parameters.
- Concurrent positions are tracked via an `open_positions` list instead of
  a single `in_position` flag.
- A cooldown of 5 trading days is enforced between opening successive
  positions.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from options.iv_model import (
    historical_volatility, generate_iv_series, iv_rank, iv_percentile,
    classify_vol_regime, generate_iv_signals, IVSignal,
    generate_pcr_series, generate_pcr_signals, generate_vix_bands,
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


# ── Strategy Mixin (shared helpers) ─────────────────────────────────────────

class _StrategyMixin:
    """Shared utility methods for all option strategies.

    Eliminates duplication of _find_strike_by_delta, _position_value, _calc_pnl
    across strategy classes.
    """

    @staticmethod
    def _find_strike_by_delta(S: float, T: float, r: float, sigma: float,
                              target_delta: float, option_type: str) -> float:
        """Find the strike price closest to a target delta.

        Searches ±40 strikes around current price using a step size appropriate
        for the price level. Returns the strike whose delta best matches target.

        Pseudocode:
            for each candidate strike K in [S - 40*step, S + 40*step]:
                d = BS_delta(S, K, T, r, sigma, type)
                if |d - target_delta| is smallest so far → save K
        """
        step = 1.0 if S < 200 else 2.0 if S < 500 else 5.0
        best_strike = S
        best_diff = float('inf')
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

    @staticmethod
    def _position_value(position: 'OptionPosition', current_price: float,
                        remaining_dte: int, current_iv: float, r: float) -> float:
        """Cost to close the position at current market prices.

        Pseudocode:
            total = 0
            for each leg:
                price = BS_price(S_now, K, T_remain, r, iv_now)
                if leg is short: total += price × contracts × 100
                if leg is long:  total -= price × contracts × 100
        """
        T = max(remaining_dte / 365.0, 1 / 365.0)
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

    @staticmethod
    def _calc_pnl(position: 'OptionPosition', current_price: float,
                  remaining_dte: int, current_iv: float, r: float) -> float:
        """P&L at exit = initial credit - cost to close."""
        close_cost = _StrategyMixin._position_value(
            position, current_price, remaining_dte, current_iv, r
        )
        return position.net_credit - close_cost

    @staticmethod
    def _long_position_value(position: 'OptionPosition', current_price: float,
                             remaining_dte: int, current_iv: float, r: float) -> float:
        """Current market value of long option legs (all legs positive)."""
        T = max(remaining_dte / 365.0, 1 / 365.0)
        total = 0.0
        for leg in position.legs:
            if leg.option_type == "call":
                price = call_price(current_price, leg.strike, T, r, current_iv)
            else:
                price = put_price(current_price, leg.strike, T, r, current_iv)
            total += price * leg.contracts * 100
        return total


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

class IronCondorStrategy(_StrategyMixin, OptionStrategy):
    """Iron Condor: Sell OTM put spread + OTM call spread.

    Best when: High IV, expecting range-bound price action.
    - Sell OTM put + Buy further OTM put (bull put spread)
    - Sell OTM call + Buy further OTM call (bear call spread)
    - Profit from time decay if price stays within the short strikes.

    Entry: IV Rank > threshold
    Exit: DTE < exit_dte, or profit target hit, or max loss hit
    """

    def __init__(self, iv_rank_entry: float = 40.0, dte_target: int = 45,
                 dte_exit: int = 21, short_delta: float = 0.16,
                 wing_width: float = 5.0, profit_target: float = 0.50,
                 stop_loss: float = 2.0, contracts: int = 1,
                 ivr_lookback: int = 60, max_concurrent: int = 3):
        self.iv_rank_entry = iv_rank_entry
        self.dte_target = dte_target
        self.dte_exit = dte_exit
        self.short_delta = short_delta  # Target delta for short strikes
        self.wing_width = wing_width    # Width between short and long strikes
        self.profit_target = profit_target  # Close at 50% of max profit
        self.stop_loss = stop_loss      # Close at 2x credit received loss
        self.contracts = contracts
        self.ivr_lookback = ivr_lookback
        self.max_concurrent = max_concurrent

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
            "ivr_lookback": self.ivr_lookback,
            "max_concurrent": self.max_concurrent,
        }

    def generate_trades(self, closes, highs, lows, dates, risk_free_rate=0.04):
        n = len(closes)
        positions = [None] * n

        hv = historical_volatility(closes, window=20)
        iv_series = generate_iv_series(closes, base_iv=0.20, seed=42)

        open_positions = []  # list of (entry_idx, position)
        last_open_idx = -999  # cooldown tracking

        for i in range(self.ivr_lookback, n):
            iv_history = iv_series[max(0, i - self.ivr_lookback):i]
            current_iv = iv_series[i]
            ivr = iv_rank(current_iv, iv_history)
            price = closes[i]
            T = self.dte_target / 365.0

            # --- Check exits for all open positions ---
            still_open = []
            for entry_idx, pos in open_positions:
                days_held = i - entry_idx
                remaining_dte = self.dte_target - days_held

                exited = False
                if remaining_dte <= self.dte_exit:
                    pos.exit_date = dates[i]
                    pnl = self._calc_pnl(pos, price, remaining_dte,
                                         iv_series[i], risk_free_rate)
                    pos.exit_pnl = pnl
                    exited = True
                else:
                    current_value = self._position_value(pos, price,
                                                         remaining_dte, iv_series[i],
                                                         risk_free_rate)
                    credit = pos.net_credit
                    if credit > 0:
                        profit_pct = (credit - current_value) / credit
                        if profit_pct >= self.profit_target:
                            pos.exit_date = dates[i]
                            pos.exit_pnl = credit - current_value
                            exited = True
                        elif current_value > credit * self.stop_loss:
                            pos.exit_date = dates[i]
                            pos.exit_pnl = credit - current_value
                            exited = True

                if not exited:
                    still_open.append((entry_idx, pos))

            open_positions = still_open

            # --- Check entry if capacity available and cooldown elapsed ---
            if len(open_positions) < self.max_concurrent and (i - last_open_idx) >= 5:
                if ivr >= self.iv_rank_entry:
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
                    open_positions.append((i, position))
                    last_open_idx = i

        return positions


# ── Vertical Spread Strategy ────────────────────────────────────────────────

class VerticalSpreadStrategy(_StrategyMixin, OptionStrategy):
    """Bull Put Spread or Bear Call Spread based on trend + IV.

    Combines directional bias with premium selling:
    - Uptrend + High IV → Bull Put Spread (sell put spread below market)
    - Downtrend + High IV → Bear Call Spread (sell call spread above market)
    """

    def __init__(self, ma_fast: int = 20, ma_slow: int = 50,
                 iv_rank_entry: float = 30.0, dte_target: int = 30,
                 dte_exit: int = 10, spread_width: float = 5.0,
                 short_delta: float = 0.30, profit_target: float = 0.65,
                 contracts: int = 1,
                 ivr_lookback: int = 60, max_concurrent: int = 3):
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.iv_rank_entry = iv_rank_entry
        self.dte_target = dte_target
        self.dte_exit = dte_exit
        self.spread_width = spread_width
        self.short_delta = short_delta
        self.profit_target = profit_target
        self.contracts = contracts
        self.ivr_lookback = ivr_lookback
        self.max_concurrent = max_concurrent

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
            "ivr_lookback": self.ivr_lookback,
            "max_concurrent": self.max_concurrent,
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

        open_positions = []
        last_open_idx = -999

        for i in range(self.ivr_lookback, n):
            iv_history = iv_series[max(0, i - self.ivr_lookback):i]
            current_iv = iv_series[i]
            ivr = iv_rank(current_iv, iv_history)
            price = closes[i]
            T = self.dte_target / 365.0

            # --- Check exits for all open positions ---
            still_open = []
            for entry_idx, pos in open_positions:
                days_held = i - entry_idx
                remaining_dte = self.dte_target - days_held

                exited = False
                if remaining_dte <= self.dte_exit:
                    pos.exit_date = dates[i]
                    pnl = self._calc_pnl(pos, price, remaining_dte,
                                         iv_series[i], risk_free_rate)
                    pos.exit_pnl = pnl
                    exited = True
                else:
                    current_value = self._position_value(pos, price,
                                                         remaining_dte, iv_series[i],
                                                         risk_free_rate)
                    credit = pos.net_credit
                    if credit > 0:
                        profit_pct = (credit - current_value) / credit
                        if profit_pct >= self.profit_target:
                            pos.exit_date = dates[i]
                            pos.exit_pnl = credit - current_value
                            exited = True

                if not exited:
                    still_open.append((entry_idx, pos))

            open_positions = still_open

            # --- Check entry ---
            if len(open_positions) < self.max_concurrent and (i - last_open_idx) >= 5:
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
                    open_positions.append((i, position))
                    last_open_idx = i

        return positions


# ── Wheel Strategy ──────────────────────────────────────────────────────────

class WheelStrategy(_StrategyMixin, OptionStrategy):
    """The Wheel: Sell CSPs → Get assigned → Sell Covered Calls → Repeat.

    One of the most popular income strategies for SPY/QQQ:
    Phase 1: Sell cash-secured puts at support levels
    Phase 2: If assigned, sell covered calls above cost basis
    Phase 3: If called away, go back to Phase 1

    Entry: IV Rank > threshold, underlying near support

    Note: The Wheel is inherently single-position (one stock lot at a time)
    so max_concurrent does not apply in the same way. However ivr_lookback
    is honoured for IV rank calculation.
    """

    def __init__(self, iv_rank_entry: float = 30.0, dte_target: int = 30,
                 put_delta: float = 0.30, call_delta: float = 0.30,
                 contracts: int = 1,
                 ivr_lookback: int = 60, max_concurrent: int = 3):
        self.iv_rank_entry = iv_rank_entry
        self.dte_target = dte_target
        self.put_delta = put_delta
        self.call_delta = call_delta
        self.contracts = contracts
        self.ivr_lookback = ivr_lookback
        self.max_concurrent = max_concurrent

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
            "ivr_lookback": self.ivr_lookback,
            "max_concurrent": self.max_concurrent,
        }

    def generate_trades(self, closes, highs, lows, dates, risk_free_rate=0.04):
        n = len(closes)
        positions = [None] * n

        hv = historical_volatility(closes, window=20)
        iv_series = generate_iv_series(closes, base_iv=0.20, seed=42)

        # State machine: "cash" → "csp_open" → "assigned" → "cc_open" → "cash"
        # The Wheel is inherently sequential (one stock lot), so we keep the
        # state-machine approach but honour ivr_lookback for the loop start.
        state = "cash"
        position_entry_idx = 0
        current_position = None
        cost_basis = 0.0  # Track cost basis when assigned

        for i in range(self.ivr_lookback, n):
            iv_history = iv_series[max(0, i - self.ivr_lookback):i]
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


# ── Straddle/Strangle Strategy ──────────────────────────────────────────────

class StraddleStrategy(_StrategyMixin, OptionStrategy):
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
                 stop_loss: float = 0.50, contracts: int = 1,
                 ivr_lookback: int = 60, max_concurrent: int = 3):
        self.iv_rank_entry = iv_rank_entry  # Enter when IV rank BELOW this
        self.dte_target = dte_target
        self.dte_exit = dte_exit
        self.use_strangle = use_strangle
        self.strangle_width = strangle_width
        self.profit_target = profit_target  # 100% profit target
        self.stop_loss = stop_loss          # Close if lost 50% of debit
        self.contracts = contracts
        self.ivr_lookback = ivr_lookback
        self.max_concurrent = max_concurrent

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
            "ivr_lookback": self.ivr_lookback,
            "max_concurrent": self.max_concurrent,
        }

    def generate_trades(self, closes, highs, lows, dates, risk_free_rate=0.04):
        n = len(closes)
        positions = [None] * n

        iv_series = generate_iv_series(closes, base_iv=0.20, seed=42)

        open_positions = []
        last_open_idx = -999

        for i in range(self.ivr_lookback, n):
            iv_history = iv_series[max(0, i - self.ivr_lookback):i]
            current_iv = iv_series[i]
            ivr = iv_rank(current_iv, iv_history)
            price = closes[i]
            T = self.dte_target / 365.0

            # --- Check exits for all open positions ---
            still_open = []
            for entry_idx, pos in open_positions:
                days_held = i - entry_idx
                remaining_dte = self.dte_target - days_held

                exited = False
                if remaining_dte <= self.dte_exit:
                    pos.exit_date = dates[i]
                    pnl = self._straddle_pnl(pos, price, remaining_dte,
                                             iv_series[i], risk_free_rate)
                    pos.exit_pnl = pnl
                    exited = True
                else:
                    current_value = self._long_position_value(pos, price,
                                                              remaining_dte, iv_series[i],
                                                              risk_free_rate)
                    debit_paid = abs(pos.net_credit)
                    if debit_paid > 0:
                        profit = current_value - debit_paid
                        if profit >= debit_paid * self.profit_target:
                            pos.exit_date = dates[i]
                            pos.exit_pnl = profit
                            exited = True
                        elif profit <= -debit_paid * self.stop_loss:
                            pos.exit_date = dates[i]
                            pos.exit_pnl = profit
                            exited = True

                if not exited:
                    still_open.append((entry_idx, pos))

            open_positions = still_open

            # --- Check entry ---
            if len(open_positions) < self.max_concurrent and (i - last_open_idx) >= 5:
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
                    open_positions.append((i, position))
                    last_open_idx = i

        return positions

    def _straddle_pnl(self, position, current_price, remaining_dte, current_iv, r):
        """P&L for long-only structures: current value - debit paid."""
        current_value = self._long_position_value(
            position, current_price, remaining_dte, current_iv, r
        )
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
                 dte_target: int = 30, contracts: int = 1,
                 ivr_lookback: int = 60, max_concurrent: int = 3):
        self.high_iv_threshold = high_iv_threshold
        self.low_iv_threshold = low_iv_threshold
        self.dte_target = dte_target
        self.contracts = contracts
        self.ivr_lookback = ivr_lookback
        self.max_concurrent = max_concurrent

        # Sub-strategies — pass through ivr_lookback and max_concurrent
        self._ic = IronCondorStrategy(
            iv_rank_entry=high_iv_threshold, dte_target=dte_target,
            contracts=contracts, ivr_lookback=ivr_lookback,
            max_concurrent=max_concurrent,
        )
        self._vs = VerticalSpreadStrategy(
            iv_rank_entry=30.0, dte_target=dte_target,
            contracts=contracts, ivr_lookback=ivr_lookback,
            max_concurrent=max_concurrent,
        )
        self._straddle = StraddleStrategy(
            iv_rank_entry=low_iv_threshold, dte_target=dte_target + 15,
            contracts=contracts, ivr_lookback=ivr_lookback,
            max_concurrent=max_concurrent,
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
            "ivr_lookback": self.ivr_lookback,
            "max_concurrent": self.max_concurrent,
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
        open_positions = []  # list of (entry_idx, dte_target)
        last_open_idx = -999

        for i in range(self.ivr_lookback, n):
            # Expire tracked positions
            open_positions = [(eidx, dte) for eidx, dte in open_positions
                              if (i - eidx) < dte]

            if len(open_positions) >= self.max_concurrent:
                continue
            if (i - last_open_idx) < 5:
                continue

            iv_history = iv_series[max(0, i - self.ivr_lookback):i]
            current_iv = iv_series[i]
            ivr = iv_rank(current_iv, iv_history)

            # Pick strategy based on IV regime
            selected = None
            selected_dte = self.dte_target
            if ivr >= self.high_iv_threshold and ic_trades[i] is not None:
                selected = ic_trades[i]
                selected_dte = self._ic.dte_target
            elif ivr <= self.low_iv_threshold and st_trades[i] is not None:
                selected = st_trades[i]
                selected_dte = self._straddle.dte_target
            elif vs_trades[i] is not None:
                selected = vs_trades[i]
                selected_dte = self._vs.dte_target

            if selected is not None:
                positions[i] = selected
                open_positions.append((i, selected_dte))
                last_open_idx = i

        return positions


# ── Butterfly Spread Strategy ────────────────────────────────────────────────

class ButterflySpreadStrategy(_StrategyMixin, OptionStrategy):
    """Long Call Butterfly: profit from price pinning at ATM strike.

    Structure:
        Buy 1 call at K - wing_width  (lower wing)
        Sell 2 calls at K             (body — ATM short)
        Buy 1 call at K + wing_width  (upper wing)

    Economics:
        Net Debit = lower_wing + upper_wing - 2 * body
        Max Profit = wing_width × 100 - debit  (at body strike at expiry)
        Max Loss   = debit paid              (price outside wings)

    Entry conditions (adapted from PyPatel volatility strategies):
        - IVR between low_iv and high_iv (moderate vol — not too rich, not too cheap)
        - Neutral price outlook: no strong trend
        - 30-45 DTE for optimal theta decay and profit window

    Exit:
        - DTE reaches exit threshold
        - Profit target hit (75% of max profit)
        - Stop loss (150% of debit paid)
    """

    def __init__(self, low_iv_threshold: float = 10.0,
                 high_iv_threshold: float = 70.0,
                 dte_target: int = 35, dte_exit: int = 10,
                 wing_width: float = 5.0,
                 profit_target: float = 0.75,
                 stop_loss_mult: float = 1.50,
                 contracts: int = 1,
                 ivr_lookback: int = 60, max_concurrent: int = 3):
        self.low_iv_threshold = low_iv_threshold
        self.high_iv_threshold = high_iv_threshold
        self.dte_target = dte_target
        self.dte_exit = dte_exit
        self.wing_width = wing_width
        self.profit_target = profit_target  # Close at 75% of max profit
        self.stop_loss_mult = stop_loss_mult  # Close if loss > 1.5× debit
        self.contracts = contracts
        self.ivr_lookback = ivr_lookback
        self.max_concurrent = max_concurrent

    @property
    def name(self) -> str:
        return "Long Call Butterfly"

    @property
    def params(self) -> Dict:
        return {
            "low_iv_threshold": self.low_iv_threshold,
            "high_iv_threshold": self.high_iv_threshold,
            "dte_target": self.dte_target,
            "wing_width": self.wing_width,
            "profit_target": self.profit_target,
            "ivr_lookback": self.ivr_lookback,
            "max_concurrent": self.max_concurrent,
        }

    def generate_trades(self, closes, highs, lows, dates, risk_free_rate=0.04):
        n = len(closes)
        positions = [None] * n

        iv_series = generate_iv_series(closes, base_iv=0.20, seed=42)

        open_positions = []
        last_open_idx = -999

        for i in range(self.ivr_lookback, n):
            iv_history = iv_series[max(0, i - self.ivr_lookback):i]
            current_iv = iv_series[i]
            ivr = iv_rank(current_iv, iv_history)
            price = closes[i]
            T = self.dte_target / 365.0

            # --- Check exits for all open positions ---
            still_open = []
            for entry_idx, pos in open_positions:
                days_held = i - entry_idx
                remaining_dte = self.dte_target - days_held

                exited = False
                if remaining_dte <= self.dte_exit:
                    pos.exit_date = dates[i]
                    pnl = self._butterfly_pnl(pos, price,
                                              remaining_dte, iv_series[i], risk_free_rate)
                    pos.exit_pnl = pnl
                    exited = True
                else:
                    current_val = self._butterfly_current_value(
                        pos, price, remaining_dte, iv_series[i], risk_free_rate
                    )
                    debit_paid = abs(pos.net_credit)
                    max_profit_est = self.wing_width * self.contracts * 100 - debit_paid

                    if debit_paid > 0 and max_profit_est > 0:
                        pnl_now = current_val - debit_paid
                        if pnl_now >= max_profit_est * self.profit_target:
                            pos.exit_date = dates[i]
                            pos.exit_pnl = pnl_now
                            exited = True
                        elif pnl_now <= -debit_paid * self.stop_loss_mult:
                            pos.exit_date = dates[i]
                            pos.exit_pnl = pnl_now
                            exited = True

                if not exited:
                    still_open.append((entry_idx, pos))

            open_positions = still_open

            # --- Check entry ---
            if len(open_positions) < self.max_concurrent and (i - last_open_idx) >= 5:
                # Entry: moderate IV regime (not too high, not too low)
                if self.low_iv_threshold <= ivr <= self.high_iv_threshold:
                    step = 1.0 if price < 200 else 2.0 if price < 500 else 5.0
                    body_strike = round(price / step) * step  # ATM
                    lower_strike = body_strike - self.wing_width
                    upper_strike = body_strike + self.wing_width

                    if lower_strike <= 0:
                        continue

                    lower_prem = call_price(price, lower_strike, T, risk_free_rate, current_iv)
                    body_prem = call_price(price, body_strike, T, risk_free_rate, current_iv)
                    upper_prem = call_price(price, upper_strike, T, risk_free_rate, current_iv)

                    # Long butterfly is ALWAYS a debit by call-price convexity:
                    #   net_debit = C(K-w) + C(K+w) - 2*C(K) >= 0
                    # Skip only if max profit is too small to be worthwhile.
                    net_debit = (lower_prem - 2 * body_prem + upper_prem) * self.contracts * 100
                    max_profit_est = self.wing_width * self.contracts * 100 - net_debit
                    if max_profit_est <= 0.01:
                        # Wing_width too narrow vs debit — no profit potential
                        continue

                    position = OptionPosition(
                        strategy_name="long_call_butterfly",
                        underlying_price=price,
                        entry_date=dates[i],
                        legs=[
                            OptionTrade("call", "buy", lower_strike, lower_prem,
                                        self.contracts, self.dte_target, current_iv,
                                        delta(price, lower_strike, T, risk_free_rate, current_iv, "call")),
                            OptionTrade("call", "sell", body_strike, body_prem,
                                        self.contracts * 2, self.dte_target, current_iv,
                                        delta(price, body_strike, T, risk_free_rate, current_iv, "call")),
                            OptionTrade("call", "buy", upper_strike, upper_prem,
                                        self.contracts, self.dte_target, current_iv,
                                        delta(price, upper_strike, T, risk_free_rate, current_iv, "call")),
                        ]
                    )

                    positions[i] = position
                    open_positions.append((i, position))
                    last_open_idx = i

        return positions

    def _butterfly_current_value(self, position, current_price, remaining_dte,
                                  current_iv, r) -> float:
        """Mark-to-market value of the butterfly (net of all legs).

        Long call butterfly:
          value = long_lower_value - 2*short_body_value + long_upper_value
        """
        T = max(remaining_dte / 365.0, 1 / 365.0)
        total = 0.0
        for leg in position.legs:
            price = call_price(current_price, leg.strike, T, r, current_iv)
            if leg.action == "buy":
                total += price * leg.contracts * 100
            else:
                total -= price * leg.contracts * 100
        return total

    def _butterfly_pnl(self, position, current_price, remaining_dte,
                        current_iv, r) -> float:
        """P&L at exit = current value - debit paid."""
        current_val = self._butterfly_current_value(
            position, current_price, remaining_dte, current_iv, r
        )
        debit_paid = abs(position.net_credit)
        return current_val - debit_paid


# ── PCR-Enhanced Iron Condor Strategy ───────────────────────────────────────

class PCREnhancedStrategy(_StrategyMixin, OptionStrategy):
    """Iron Condor with PCR + VIX-band dual-confirmation entry filter.

    Enhances the standard Iron Condor with two additional market-sentiment
    signals inspired by PyPatel's VIX_Strategy.py and PCR_strategy.py:

    1. PCR (Put/Call Ratio) signal:
       Entry only when PCR is ABOVE its upper Bollinger Band.
       Extreme fear (high PCR) → IV is elevated → optimal time to sell premium.

    2. VIX-band signal (using synthetic IV series as VIX proxy):
       Entry only when current IV is ABOVE its 20-day mean + 1.5σ.
       IV spike → mean-reversion edge → sell premium before vol crush.

    Combined entry requirements (gates are optional with relaxed defaults):
        IVR >= iv_rank_entry          (standard IV rank gate — always on)
        PCR >= PCR upper band         (optional, default OFF)
        IV  >= VIX upper band         (optional, default OFF)

    Exit: Same as Iron Condor (DTE, profit target, stop loss).
    """

    def __init__(self, iv_rank_entry: float = 40.0,
                 dte_target: int = 45, dte_exit: int = 21,
                 short_delta: float = 0.16, wing_width: float = 5.0,
                 profit_target: float = 0.50, stop_loss: float = 2.0,
                 pcr_window: int = 20, vix_window: int = 20,
                 require_pcr: bool = False, require_vix_band: bool = False,
                 contracts: int = 1,
                 ivr_lookback: int = 60, max_concurrent: int = 3):
        self.iv_rank_entry = iv_rank_entry
        self.dte_target = dte_target
        self.dte_exit = dte_exit
        self.short_delta = short_delta
        self.wing_width = wing_width
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.pcr_window = pcr_window
        self.vix_window = vix_window
        self.require_pcr = require_pcr
        self.require_vix_band = require_vix_band
        self.contracts = contracts
        self.ivr_lookback = ivr_lookback
        self.max_concurrent = max_concurrent

    @property
    def name(self) -> str:
        return "PCR-Enhanced Iron Condor"

    @property
    def params(self) -> Dict:
        return {
            "iv_rank_entry": self.iv_rank_entry,
            "dte_target": self.dte_target,
            "short_delta": self.short_delta,
            "wing_width": self.wing_width,
            "require_pcr": self.require_pcr,
            "require_vix_band": self.require_vix_band,
            "ivr_lookback": self.ivr_lookback,
            "max_concurrent": self.max_concurrent,
        }

    def generate_trades(self, closes, highs, lows, dates, risk_free_rate=0.04):
        n = len(closes)
        positions = [None] * n

        iv_series = generate_iv_series(closes, base_iv=0.20, seed=42)
        pcr_series = generate_pcr_series(closes, iv_series, seed=42)
        pcr_signals = generate_pcr_signals(pcr_series, window=self.pcr_window)
        vix_bands = generate_vix_bands(iv_series, window=self.vix_window)

        open_positions = []
        last_open_idx = -999

        for i in range(self.ivr_lookback, n):
            iv_history = iv_series[max(0, i - self.ivr_lookback):i]
            current_iv = iv_series[i]
            ivr = iv_rank(current_iv, iv_history)
            price = closes[i]
            T = self.dte_target / 365.0

            # --- Check exits for all open positions ---
            still_open = []
            for entry_idx, pos in open_positions:
                days_held = i - entry_idx
                remaining_dte = self.dte_target - days_held

                exited = False
                if remaining_dte <= self.dte_exit:
                    pos.exit_date = dates[i]
                    pnl = self._calc_pnl(pos, price, remaining_dte,
                                         iv_series[i], risk_free_rate)
                    pos.exit_pnl = pnl
                    exited = True
                else:
                    current_value = self._position_value(
                        pos, price, remaining_dte, iv_series[i], risk_free_rate
                    )
                    credit = pos.net_credit
                    if credit > 0:
                        profit_pct = (credit - current_value) / credit
                        if profit_pct >= self.profit_target:
                            pos.exit_date = dates[i]
                            pos.exit_pnl = credit - current_value
                            exited = True
                        elif current_value > credit * self.stop_loss:
                            pos.exit_date = dates[i]
                            pos.exit_pnl = credit - current_value
                            exited = True

                if not exited:
                    still_open.append((entry_idx, pos))

            open_positions = still_open

            # --- Check entry ---
            if len(open_positions) < self.max_concurrent and (i - last_open_idx) >= 5:
                # Gate 1: Standard IV rank
                if ivr < self.iv_rank_entry:
                    continue

                # Gate 2: PCR confirmation (optional — default OFF)
                if self.require_pcr and pcr_signals[i].signal != 1:
                    continue

                # Gate 3: VIX band (optional — default OFF)
                vix_mean, vix_upper, vix_lower = vix_bands[i]
                if self.require_vix_band and current_iv < vix_upper:
                    continue

                # All active gates passed → build Iron Condor
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

                short_put_prem = put_price(price, put_strike, T, risk_free_rate, current_iv)
                long_put_prem = put_price(price, put_long_strike, T, risk_free_rate, current_iv)
                short_call_prem = call_price(price, call_strike, T, risk_free_rate, current_iv)
                long_call_prem = call_price(price, call_long_strike, T, risk_free_rate, current_iv)

                position = OptionPosition(
                    strategy_name="pcr_enhanced_ic",
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
                open_positions.append((i, position))
                last_open_idx = i

        return positions


# ── TA-Driven Strategy ──────────────────────────────────────────────────────

class TADrivenStrategy(_StrategyMixin, OptionStrategy):
    """Technical-Analysis-driven option strategy.

    Uses MACD (12/26/9), RSI (14), and Bollinger Bands (20, 2sigma) on the
    underlying price to generate option trade signals.  No IV-rank gate --
    entries are purely TA-driven.

    Signal logic:
        1. MACD bullish cross + RSI < 70 + price near lower BB
           → Bull Put Spread (credit)
        2. MACD bearish cross + RSI > 30 + price near upper BB
           → Bear Call Spread (credit)
        3. RSI < 25 (deep oversold)
           → Buy Call (debit)
        4. RSI > 75 (deep overbought)
           → Buy Put (debit)
        5. Bollinger squeeze (bandwidth < 20th percentile of last 100 days)
           + any MACD cross on that day
           → Long Straddle (debit)

    Exit mechanics: DTE exit, profit target, stop loss — same as other
    strategies.
    """

    def __init__(self, dte_target: int = 30, dte_exit: int = 10,
                 spread_width: float = 5.0, short_delta: float = 0.30,
                 profit_target: float = 0.65, stop_loss: float = 0.50,
                 contracts: int = 1,
                 # MACD parameters
                 macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
                 # RSI parameters
                 rsi_period: int = 14,
                 # Bollinger Band parameters
                 bb_period: int = 20, bb_std: float = 2.0,
                 # "near band" threshold — fraction of band width
                 bb_near_pct: float = 0.10,
                 # Squeeze percentile
                 squeeze_percentile: float = 20.0,
                 ivr_lookback: int = 60, max_concurrent: int = 3):
        self.dte_target = dte_target
        self.dte_exit = dte_exit
        self.spread_width = spread_width
        self.short_delta = short_delta
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.contracts = contracts
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.bb_near_pct = bb_near_pct
        self.squeeze_percentile = squeeze_percentile
        self.ivr_lookback = ivr_lookback
        self.max_concurrent = max_concurrent

    @property
    def name(self) -> str:
        return "TA-Driven"

    @property
    def params(self) -> Dict:
        return {
            "dte_target": self.dte_target,
            "spread_width": self.spread_width,
            "short_delta": self.short_delta,
            "profit_target": self.profit_target,
            "stop_loss": self.stop_loss,
            "macd_fast": self.macd_fast,
            "macd_slow": self.macd_slow,
            "macd_signal": self.macd_signal,
            "rsi_period": self.rsi_period,
            "bb_period": self.bb_period,
            "bb_std": self.bb_std,
            "ivr_lookback": self.ivr_lookback,
            "max_concurrent": self.max_concurrent,
        }

    # ── Technical indicator helpers ──────────────────────────────────────

    @staticmethod
    def _ema(values: List[float], period: int) -> List[float]:
        """Exponential Moving Average."""
        result = [0.0] * len(values)
        if len(values) == 0 or period <= 0:
            return result
        k = 2.0 / (period + 1)
        result[0] = values[0]
        for i in range(1, len(values)):
            result[i] = values[i] * k + result[i - 1] * (1 - k)
        return result

    def _compute_macd(self, closes: List[float]) -> Tuple[List[float], List[float], List[float]]:
        """Return (macd_line, signal_line, histogram)."""
        ema_fast = self._ema(closes, self.macd_fast)
        ema_slow = self._ema(closes, self.macd_slow)
        macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
        signal_line = self._ema(macd_line, self.macd_signal)
        histogram = [m - s for m, s in zip(macd_line, signal_line)]
        return macd_line, signal_line, histogram

    def _compute_rsi(self, closes: List[float]) -> List[float]:
        """Relative Strength Index (Wilder's smoothed)."""
        n = len(closes)
        rsi = [50.0] * n  # neutral default
        if n < self.rsi_period + 1:
            return rsi

        # Initial average gain/loss over first rsi_period
        gains = 0.0
        losses = 0.0
        for j in range(1, self.rsi_period + 1):
            change = closes[j] - closes[j - 1]
            if change > 0:
                gains += change
            else:
                losses -= change  # make positive

        avg_gain = gains / self.rsi_period
        avg_loss = losses / self.rsi_period

        if avg_loss == 0:
            rsi[self.rsi_period] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[self.rsi_period] = 100.0 - 100.0 / (1.0 + rs)

        for j in range(self.rsi_period + 1, n):
            change = closes[j] - closes[j - 1]
            gain = max(change, 0.0)
            loss = max(-change, 0.0)
            avg_gain = (avg_gain * (self.rsi_period - 1) + gain) / self.rsi_period
            avg_loss = (avg_loss * (self.rsi_period - 1) + loss) / self.rsi_period
            if avg_loss == 0:
                rsi[j] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[j] = 100.0 - 100.0 / (1.0 + rs)

        return rsi

    def _compute_bollinger(self, closes: List[float]) -> Tuple[List[float], List[float], List[float], List[float]]:
        """Return (middle_band, upper_band, lower_band, bandwidth)."""
        n = len(closes)
        mid = [0.0] * n
        upper = [0.0] * n
        lower = [0.0] * n
        bandwidth = [0.0] * n

        for i in range(self.bb_period - 1, n):
            window = closes[i - self.bb_period + 1:i + 1]
            mean = sum(window) / self.bb_period
            var = sum((x - mean) ** 2 for x in window) / self.bb_period
            std = math.sqrt(var) if var > 0 else 0.0
            mid[i] = mean
            upper[i] = mean + self.bb_std * std
            lower[i] = mean - self.bb_std * std
            bandwidth[i] = (upper[i] - lower[i]) / mean if mean > 0 else 0.0

        return mid, upper, lower, bandwidth

    # ── Main trade generation ────────────────────────────────────────────

    def generate_trades(self, closes, highs, lows, dates, risk_free_rate=0.04):
        n = len(closes)
        positions = [None] * n

        iv_series = generate_iv_series(closes, base_iv=0.20, seed=42)

        # Pre-compute indicators
        macd_line, signal_line, histogram = self._compute_macd(closes)
        rsi = self._compute_rsi(closes)
        bb_mid, bb_upper, bb_lower, bb_bw = self._compute_bollinger(closes)

        open_positions = []  # list of (entry_idx, position, position_type)
        last_open_idx = -999

        # Minimum warm-up: need enough bars for all indicators
        min_start = max(self.macd_slow + self.macd_signal, self.rsi_period + 1,
                        self.bb_period, self.ivr_lookback)

        for i in range(min_start, n):
            price = closes[i]
            current_iv = iv_series[i]
            T = self.dte_target / 365.0

            # --- Check exits for all open positions ---
            still_open = []
            for entry_idx, pos, pos_type in open_positions:
                days_held = i - entry_idx
                remaining_dte = self.dte_target - days_held

                exited = False
                if remaining_dte <= self.dte_exit:
                    pos.exit_date = dates[i]
                    if pos_type in ("long_call", "long_put", "long_straddle"):
                        pnl = self._long_position_value(pos, price, remaining_dte,
                                                        iv_series[i], risk_free_rate) - abs(pos.net_credit)
                    else:
                        pnl = self._calc_pnl(pos, price, remaining_dte,
                                             iv_series[i], risk_free_rate)
                    pos.exit_pnl = pnl
                    exited = True
                else:
                    if pos_type in ("long_call", "long_put", "long_straddle"):
                        current_value = self._long_position_value(pos, price,
                                                                   remaining_dte, iv_series[i],
                                                                   risk_free_rate)
                        debit_paid = abs(pos.net_credit)
                        if debit_paid > 0:
                            profit = current_value - debit_paid
                            if profit >= debit_paid * self.profit_target:
                                pos.exit_date = dates[i]
                                pos.exit_pnl = profit
                                exited = True
                            elif profit <= -debit_paid * self.stop_loss:
                                pos.exit_date = dates[i]
                                pos.exit_pnl = profit
                                exited = True
                    else:
                        # Credit spread exit
                        current_value = self._position_value(pos, price,
                                                              remaining_dte, iv_series[i],
                                                              risk_free_rate)
                        credit = pos.net_credit
                        if credit > 0:
                            profit_pct = (credit - current_value) / credit
                            if profit_pct >= self.profit_target:
                                pos.exit_date = dates[i]
                                pos.exit_pnl = credit - current_value
                                exited = True
                            elif current_value > credit * 2.0:
                                pos.exit_date = dates[i]
                                pos.exit_pnl = credit - current_value
                                exited = True

                if not exited:
                    still_open.append((entry_idx, pos, pos_type))

            open_positions = still_open

            # --- Check entry ---
            if len(open_positions) >= self.max_concurrent:
                continue
            if (i - last_open_idx) < 5:
                continue

            # Detect MACD crosses
            macd_bullish_cross = (histogram[i] > 0 and histogram[i - 1] <= 0)
            macd_bearish_cross = (histogram[i] < 0 and histogram[i - 1] >= 0)
            macd_any_cross = macd_bullish_cross or macd_bearish_cross

            # "Near" band detection
            bb_width = bb_upper[i] - bb_lower[i]
            near_lower = (price - bb_lower[i]) <= bb_width * self.bb_near_pct if bb_width > 0 else False
            near_upper = (bb_upper[i] - price) <= bb_width * self.bb_near_pct if bb_width > 0 else False

            # Bollinger squeeze detection
            is_squeeze = False
            if i >= 100:
                recent_bw = bb_bw[i - 99:i + 1]
                sorted_bw = sorted(recent_bw)
                pct_idx = max(0, int(len(sorted_bw) * self.squeeze_percentile / 100.0) - 1)
                threshold = sorted_bw[pct_idx]
                is_squeeze = bb_bw[i] <= threshold

            # --- Signal priority (first match wins) ---
            position = None
            pos_type = None

            # Signal 5: Bollinger squeeze + MACD cross → Long Straddle
            if is_squeeze and macd_any_cross:
                step = 1.0 if price < 200 else 2.0 if price < 500 else 5.0
                atm_strike = round(price / step) * step
                call_prem = call_price(price, atm_strike, T, risk_free_rate, current_iv)
                put_prem = put_price(price, atm_strike, T, risk_free_rate, current_iv)
                position = OptionPosition(
                    strategy_name="long_straddle",
                    underlying_price=price,
                    entry_date=dates[i],
                    legs=[
                        OptionTrade("call", "buy", atm_strike, call_prem,
                                    self.contracts, self.dte_target, current_iv,
                                    delta(price, atm_strike, T, risk_free_rate, current_iv, "call")),
                        OptionTrade("put", "buy", atm_strike, put_prem,
                                    self.contracts, self.dte_target, current_iv,
                                    delta(price, atm_strike, T, risk_free_rate, current_iv, "put")),
                    ]
                )
                pos_type = "long_straddle"

            # Signal 3: RSI < 25 → Buy Call
            elif rsi[i] < 25:
                step = 1.0 if price < 200 else 2.0 if price < 500 else 5.0
                atm_strike = round(price / step) * step
                prem = call_price(price, atm_strike, T, risk_free_rate, current_iv)
                position = OptionPosition(
                    strategy_name="long_call",
                    underlying_price=price,
                    entry_date=dates[i],
                    legs=[
                        OptionTrade("call", "buy", atm_strike, prem,
                                    self.contracts, self.dte_target, current_iv,
                                    delta(price, atm_strike, T, risk_free_rate, current_iv, "call")),
                    ]
                )
                pos_type = "long_call"

            # Signal 4: RSI > 75 → Buy Put
            elif rsi[i] > 75:
                step = 1.0 if price < 200 else 2.0 if price < 500 else 5.0
                atm_strike = round(price / step) * step
                prem = put_price(price, atm_strike, T, risk_free_rate, current_iv)
                position = OptionPosition(
                    strategy_name="long_put",
                    underlying_price=price,
                    entry_date=dates[i],
                    legs=[
                        OptionTrade("put", "buy", atm_strike, prem,
                                    self.contracts, self.dte_target, current_iv,
                                    delta(price, atm_strike, T, risk_free_rate, current_iv, "put")),
                    ]
                )
                pos_type = "long_put"

            # Signal 1: MACD bullish cross + RSI < 70 + near lower BB → Bull Put Spread
            elif macd_bullish_cross and rsi[i] < 70 and near_lower:
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
                pos_type = "bull_put_spread"

            # Signal 2: MACD bearish cross + RSI > 30 + near upper BB → Bear Call Spread
            elif macd_bearish_cross and rsi[i] > 30 and near_upper:
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
                pos_type = "bear_call_spread"

            if position is not None:
                positions[i] = position
                open_positions.append((i, position, pos_type))
                last_open_idx = i

        return positions
