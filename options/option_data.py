"""
Option Data Models and Synthetic Option Chain Generator

Provides data structures for option contracts and chains,
plus synthetic generation of realistic QQQ/SPY option chains
with volatility smile and term structure.
"""

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict

from options.pricing import call_price, put_price, implied_volatility, all_greeks, GreeksResult


@dataclass
class OptionContract:
    """Represents a single option contract."""
    symbol: str           # Underlying symbol (e.g., "SPY", "QQQ")
    strike: float         # Strike price
    expiration: str       # Expiration date string (YYYY-MM-DD)
    option_type: str      # "call" or "put"
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    volume: int = 0
    open_interest: int = 0
    iv: float = 0.0       # Implied volatility
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def moneyness(self) -> str:
        """Determined at chain generation time based on underlying price."""
        return self._moneyness if hasattr(self, '_moneyness') else "unknown"

    def intrinsic_value(self, underlying_price: float) -> float:
        if self.option_type == "call":
            return max(underlying_price - self.strike, 0.0)
        else:
            return max(self.strike - underlying_price, 0.0)

    def __repr__(self):
        return (f"{self.symbol} {self.expiration} {self.strike:.0f}"
                f" {self.option_type.upper()} @ {self.mid:.2f}"
                f" (IV={self.iv:.1%}, Δ={self.delta:+.3f})")


@dataclass
class OptionChain:
    """Collection of option contracts for a given expiration."""
    symbol: str
    underlying_price: float
    date: str                    # Current date
    expiration: str              # Expiration date
    dte: int                     # Days to expiration
    calls: List[OptionContract] = field(default_factory=list)
    puts: List[OptionContract] = field(default_factory=list)

    @property
    def strikes(self) -> List[float]:
        return sorted(set(c.strike for c in self.calls))

    def get_atm_strike(self) -> float:
        """Get the strike closest to the underlying price."""
        strikes = self.strikes
        if not strikes:
            return self.underlying_price
        return min(strikes, key=lambda k: abs(k - self.underlying_price))

    def get_call(self, strike: float) -> Optional[OptionContract]:
        for c in self.calls:
            if abs(c.strike - strike) < 0.01:
                return c
        return None

    def get_put(self, strike: float) -> Optional[OptionContract]:
        for p in self.puts:
            if abs(p.strike - strike) < 0.01:
                return p
        return None

    def get_otm_puts(self) -> List[OptionContract]:
        """Get out-of-the-money puts (strike < underlying)."""
        return sorted([p for p in self.puts if p.strike < self.underlying_price],
                      key=lambda p: p.strike, reverse=True)

    def get_otm_calls(self) -> List[OptionContract]:
        """Get out-of-the-money calls (strike > underlying)."""
        return sorted([c for c in self.calls if c.strike > self.underlying_price],
                      key=lambda c: c.strike)

    def __repr__(self):
        return (f"OptionChain({self.symbol} @ {self.underlying_price:.2f}, "
                f"exp={self.expiration}, DTE={self.dte}, "
                f"{len(self.calls)}C/{len(self.puts)}P)")


def _vol_smile(moneyness_ratio: float, base_vol: float, skew: float = 0.12,
               smile: float = 0.08) -> float:
    """Generate realistic implied volatility with smile/skew.

    Args:
        moneyness_ratio: strike / underlying_price
        base_vol: ATM implied volatility
        skew: negative skew factor (puts > calls in vol)
        smile: curvature factor (wings higher vol)

    Returns:
        IV for this strike with smile applied
    """
    log_m = math.log(moneyness_ratio)
    # Quadratic smile + linear skew
    iv = base_vol + smile * log_m ** 2 - skew * log_m
    return max(iv, 0.05)  # floor at 5%


def generate_option_chain(
    symbol: str,
    underlying_price: float,
    current_date: str,
    expiration_date: str,
    base_iv: float = 0.20,
    risk_free_rate: float = 0.04,
    strike_step: Optional[float] = None,
    num_strikes: int = 15,
    skew: float = 0.12,
    seed: Optional[int] = None,
) -> OptionChain:
    """Generate a realistic synthetic option chain.

    Args:
        symbol: Underlying ticker
        underlying_price: Current price
        current_date: Current date (YYYY-MM-DD)
        expiration_date: Expiration date (YYYY-MM-DD)
        base_iv: ATM implied volatility
        risk_free_rate: Risk-free rate
        strike_step: Dollar distance between strikes (auto if None)
        num_strikes: Number of strikes on each side of ATM
        skew: Volatility skew factor
        seed: Random seed for reproducibility

    Returns:
        OptionChain with calls and puts
    """
    if seed is not None:
        random.seed(seed)

    # Calculate DTE
    d_curr = datetime.strptime(current_date, "%Y-%m-%d")
    d_exp = datetime.strptime(expiration_date, "%Y-%m-%d")
    dte = (d_exp - d_curr).days
    T = dte / 365.0

    # Auto strike step based on price level
    if strike_step is None:
        if underlying_price > 400:
            strike_step = 5.0
        elif underlying_price > 100:
            strike_step = 2.0
        elif underlying_price > 50:
            strike_step = 1.0
        else:
            strike_step = 0.5

    # Generate strikes centered around ATM
    atm_strike = round(underlying_price / strike_step) * strike_step
    strikes = [atm_strike + i * strike_step
               for i in range(-num_strikes, num_strikes + 1)]
    strikes = [s for s in strikes if s > 0]

    calls = []
    puts = []

    for strike in strikes:
        moneyness = strike / underlying_price
        iv = _vol_smile(moneyness, base_iv, skew=skew)

        # Add small random noise to IV (realistic)
        iv += random.gauss(0, 0.005)
        iv = max(iv, 0.05)

        # Price using Black-Scholes
        c_price = call_price(underlying_price, strike, T, risk_free_rate, iv)
        p_price = put_price(underlying_price, strike, T, risk_free_rate, iv)

        # Generate realistic bid-ask spread (wider for OTM, tighter for ATM)
        distance = abs(moneyness - 1.0)
        base_spread = 0.02 + distance * 0.15
        c_spread = max(0.01, c_price * base_spread)
        p_spread = max(0.01, p_price * base_spread)

        # Volume and OI (higher near ATM)
        vol_factor = max(0.05, 1.0 - 3.0 * distance)
        base_volume = random.randint(100, 5000)
        base_oi = random.randint(1000, 50000)

        # Calculate Greeks
        from options.pricing import delta as calc_delta, gamma as calc_gamma
        from options.pricing import theta as calc_theta, vega as calc_vega

        # Call contract
        c = OptionContract(
            symbol=symbol,
            strike=strike,
            expiration=expiration_date,
            option_type="call",
            bid=max(0.01, c_price - c_spread / 2),
            ask=c_price + c_spread / 2,
            last=c_price + random.gauss(0, c_spread * 0.2),
            volume=int(base_volume * vol_factor),
            open_interest=int(base_oi * vol_factor),
            iv=iv,
            delta=calc_delta(underlying_price, strike, T, risk_free_rate, iv, "call"),
            gamma=calc_gamma(underlying_price, strike, T, risk_free_rate, iv),
            theta=calc_theta(underlying_price, strike, T, risk_free_rate, iv, "call"),
            vega=calc_vega(underlying_price, strike, T, risk_free_rate, iv),
        )
        calls.append(c)

        # Put contract
        p = OptionContract(
            symbol=symbol,
            strike=strike,
            expiration=expiration_date,
            option_type="put",
            bid=max(0.01, p_price - p_spread / 2),
            ask=p_price + p_spread / 2,
            last=p_price + random.gauss(0, p_spread * 0.2),
            volume=int(base_volume * vol_factor * 0.8),
            open_interest=int(base_oi * vol_factor * 0.9),
            iv=iv,
            delta=calc_delta(underlying_price, strike, T, risk_free_rate, iv, "put"),
            gamma=calc_gamma(underlying_price, strike, T, risk_free_rate, iv),
            theta=calc_theta(underlying_price, strike, T, risk_free_rate, iv, "put"),
            vega=calc_vega(underlying_price, strike, T, risk_free_rate, iv),
        )
        puts.append(p)

    return OptionChain(
        symbol=symbol,
        underlying_price=underlying_price,
        date=current_date,
        expiration=expiration_date,
        dte=dte,
        calls=sorted(calls, key=lambda c: c.strike),
        puts=sorted(puts, key=lambda p: p.strike),
    )


def generate_spy_data(days: int = 500, start_price: float = 480.0,
                      volatility: float = 0.015, trend: float = 0.0004,
                      seed: int = 42) -> 'MarketData':
    """Generate synthetic SPY price data."""
    from data.market_data import generate_synthetic_data
    return generate_synthetic_data(
        symbol="SPY", days=days, start_price=start_price,
        volatility=volatility, trend=trend, seed=seed
    )


def generate_qqq_data(days: int = 500, start_price: float = 420.0,
                      volatility: float = 0.020, trend: float = 0.0005,
                      seed: int = 42) -> 'MarketData':
    """Generate synthetic QQQ price data (higher vol than SPY)."""
    from data.market_data import generate_synthetic_data
    return generate_synthetic_data(
        symbol="QQQ", days=days, start_price=start_price,
        volatility=volatility, trend=trend, seed=seed
    )
