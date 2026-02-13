"""
Implied Volatility Analysis Module

Provides tools for analyzing volatility:
- Historical Volatility (HV) calculation
- IV Rank and IV Percentile
- Volatility regime detection
- Signal generation based on IV mean reversion
"""

import math
from typing import List, Tuple, Optional
from dataclasses import dataclass


def historical_volatility(closes: List[float], window: int = 20) -> List[float]:
    """Calculate rolling historical volatility (annualized).

    Uses log returns and annualizes by sqrt(252).

    Args:
        closes: List of closing prices
        window: Rolling window size (default 20 = ~1 month)

    Returns:
        List of HV values (same length as closes, NaN-like 0.0 for initial period)
    """
    n = len(closes)
    hv = [0.0] * n

    # Calculate log returns
    log_returns = [0.0] * n
    for i in range(1, n):
        if closes[i - 1] > 0:
            log_returns[i] = math.log(closes[i] / closes[i - 1])

    for i in range(window, n):
        returns_window = log_returns[i - window + 1: i + 1]
        mean_ret = sum(returns_window) / len(returns_window)
        variance = sum((r - mean_ret) ** 2 for r in returns_window) / (len(returns_window) - 1)
        hv[i] = math.sqrt(variance * 252)

    return hv


def realized_volatility(closes: List[float], window: int = 20) -> List[float]:
    """Alias for historical_volatility using close-to-close method."""
    return historical_volatility(closes, window)


def parkinson_volatility(highs: List[float], lows: List[float],
                         window: int = 20) -> List[float]:
    """Parkinson volatility estimator using high-low range.

    More efficient than close-to-close as it uses intraday information.
    """
    n = len(highs)
    pv = [0.0] * n
    factor = 1.0 / (4.0 * math.log(2.0))

    for i in range(window, n):
        total = 0.0
        for j in range(i - window + 1, i + 1):
            if lows[j] > 0:
                hl_ratio = highs[j] / lows[j]
                total += math.log(hl_ratio) ** 2
        pv[i] = math.sqrt(factor * total / window * 252)

    return pv


def iv_rank(current_iv: float, iv_history: List[float]) -> float:
    """IV Rank: where current IV sits relative to its 52-week range.

    IV Rank = (Current IV - 52w Low IV) / (52w High IV - 52w Low IV) * 100

    Returns:
        Percentage 0-100. High = IV is near its yearly high.
    """
    if not iv_history:
        return 50.0
    iv_min = min(iv_history)
    iv_max = max(iv_history)
    if iv_max == iv_min:
        return 50.0
    return ((current_iv - iv_min) / (iv_max - iv_min)) * 100.0


def iv_percentile(current_iv: float, iv_history: List[float]) -> float:
    """IV Percentile: percentage of days in the past year with lower IV.

    IV Percentile = (Days with IV < Current IV) / Total Days * 100

    Returns:
        Percentage 0-100. High = IV is higher than most days.
    """
    if not iv_history:
        return 50.0
    days_below = sum(1 for iv in iv_history if iv < current_iv)
    return (days_below / len(iv_history)) * 100.0


@dataclass
class VolRegime:
    """Volatility regime classification."""
    regime: str         # "low", "normal", "high", "extreme"
    hv_current: float   # Current HV
    hv_mean: float      # Long-term mean HV
    hv_std: float       # HV standard deviation
    z_score: float      # Z-score of current HV
    iv_rank_val: float  # IV Rank
    iv_pctile: float    # IV Percentile

    @property
    def is_high_vol(self) -> bool:
        return self.regime in ("high", "extreme")

    @property
    def is_low_vol(self) -> bool:
        return self.regime == "low"

    def __repr__(self):
        return (f"VolRegime({self.regime.upper()}: HV={self.hv_current:.1%}, "
                f"z={self.z_score:+.2f}, IVR={self.iv_rank_val:.0f}, "
                f"IVP={self.iv_pctile:.0f})")


def classify_vol_regime(hv_series: List[float], lookback: int = 252) -> VolRegime:
    """Classify current volatility regime.

    Args:
        hv_series: Historical volatility time series
        lookback: Period for calculating mean/std (default 252 = 1 year)

    Returns:
        VolRegime with classification
    """
    # Use the most recent non-zero values
    recent = [v for v in hv_series[-lookback:] if v > 0]
    if not recent:
        return VolRegime("normal", 0.0, 0.0, 0.0, 0.0, 50.0, 50.0)

    current_hv = recent[-1]
    mean_hv = sum(recent) / len(recent)
    std_hv = math.sqrt(sum((v - mean_hv) ** 2 for v in recent) / max(len(recent) - 1, 1))

    z_score = (current_hv - mean_hv) / std_hv if std_hv > 0 else 0.0

    ivr = iv_rank(current_hv, recent)
    ivp = iv_percentile(current_hv, recent)

    # Classify
    if z_score > 2.0:
        regime = "extreme"
    elif z_score > 1.0:
        regime = "high"
    elif z_score < -1.0:
        regime = "low"
    else:
        regime = "normal"

    return VolRegime(
        regime=regime,
        hv_current=current_hv,
        hv_mean=mean_hv,
        hv_std=std_hv,
        z_score=z_score,
        iv_rank_val=ivr,
        iv_pctile=ivp,
    )


# ── Synthetic IV Series Generation ──────────────────────────────────────────

def generate_iv_series(closes: List[float], base_iv: float = 0.20,
                       iv_mean_reversion: float = 0.05,
                       iv_vol_of_vol: float = 0.15,
                       leverage_effect: float = -0.5,
                       seed: Optional[int] = None) -> List[float]:
    """Generate a synthetic IV time series that behaves realistically.

    Key properties:
    - Mean-reverting around base_iv
    - Negatively correlated with returns (leverage effect)
    - Has its own stochastic component (vol-of-vol)
    - Clusters (high vol tends to stay high)

    Args:
        closes: Underlying closing prices
        base_iv: Long-term mean IV
        iv_mean_reversion: Speed of mean reversion (higher = faster)
        iv_vol_of_vol: Volatility of volatility
        leverage_effect: Correlation between returns and IV changes
        seed: Random seed

    Returns:
        List of IV values, one per day
    """
    import random as rng
    if seed is not None:
        rng.seed(seed)

    n = len(closes)
    iv_series = [base_iv] * n

    for i in range(1, n):
        prev_iv = iv_series[i - 1]

        # Mean reversion pull
        mr_pull = iv_mean_reversion * (base_iv - prev_iv) / 252.0

        # Return-based component (leverage effect)
        if closes[i - 1] > 0:
            ret = (closes[i] - closes[i - 1]) / closes[i - 1]
        else:
            ret = 0.0
        leverage = leverage_effect * ret * 2.0  # Amplified

        # Random shock
        shock = rng.gauss(0, iv_vol_of_vol / math.sqrt(252))

        # Update IV
        new_iv = prev_iv + mr_pull + leverage + shock
        iv_series[i] = max(0.05, min(new_iv, 1.0))  # clamp 5% - 100%

    return iv_series


# ── Trading Signals Based on IV ─────────────────────────────────────────────

@dataclass
class IVSignal:
    """Trading signal generated from IV analysis."""
    signal: int          # 1 = sell premium, -1 = buy premium, 0 = hold
    strategy_hint: str   # Suggested strategy type
    confidence: float    # 0-1 confidence level
    reason: str          # Explanation

    def __repr__(self):
        actions = {1: "SELL_PREMIUM", -1: "BUY_PREMIUM", 0: "HOLD"}
        return (f"IVSignal({actions[self.signal]}: {self.strategy_hint}, "
                f"conf={self.confidence:.0%}, {self.reason})")


def generate_iv_signals(iv_series: List[float], hv_series: List[float],
                        lookback: int = 252) -> List[IVSignal]:
    """Generate trading signals based on IV analysis.

    Logic:
    - High IV Rank (>70) + IV > HV → Sell premium (Iron Condor, Credit Spreads)
    - Low IV Rank (<30) + IV < HV → Buy premium (Straddles, Long options)
    - Extreme IV (>90 rank) → Aggressive premium selling
    - IV near HV, normal rank → No clear edge, hold

    Args:
        iv_series: Implied volatility time series
        hv_series: Historical volatility time series
        lookback: Days for IV rank calculation

    Returns:
        List of IVSignal, one per day
    """
    n = len(iv_series)
    signals = []

    for i in range(n):
        if i < lookback:
            signals.append(IVSignal(0, "none", 0.0, "Insufficient data"))
            continue

        current_iv = iv_series[i]
        current_hv = hv_series[i] if hv_series[i] > 0 else iv_series[i]
        iv_history = iv_series[max(0, i - lookback):i]

        ivr = iv_rank(current_iv, iv_history)
        ivp = iv_percentile(current_iv, iv_history)
        iv_hv_ratio = current_iv / current_hv if current_hv > 0 else 1.0

        # Decision logic
        if ivr > 80 and iv_hv_ratio > 1.2:
            signals.append(IVSignal(
                signal=1,
                strategy_hint="iron_condor",
                confidence=min(0.95, ivr / 100.0),
                reason=f"IV Rank {ivr:.0f}, IV/HV={iv_hv_ratio:.2f}: Rich premium"
            ))
        elif ivr > 60 and iv_hv_ratio > 1.1:
            signals.append(IVSignal(
                signal=1,
                strategy_hint="credit_spread",
                confidence=min(0.80, ivr / 100.0),
                reason=f"IV Rank {ivr:.0f}, IV/HV={iv_hv_ratio:.2f}: Elevated premium"
            ))
        elif ivr < 20 and iv_hv_ratio < 0.9:
            signals.append(IVSignal(
                signal=-1,
                strategy_hint="long_straddle",
                confidence=min(0.80, (100 - ivr) / 100.0),
                reason=f"IV Rank {ivr:.0f}, IV/HV={iv_hv_ratio:.2f}: Cheap premium"
            ))
        elif ivr < 30 and iv_hv_ratio < 1.0:
            signals.append(IVSignal(
                signal=-1,
                strategy_hint="long_options",
                confidence=min(0.65, (100 - ivr) / 100.0),
                reason=f"IV Rank {ivr:.0f}, IV/HV={iv_hv_ratio:.2f}: Below-average premium"
            ))
        else:
            signals.append(IVSignal(
                signal=0,
                strategy_hint="none",
                confidence=0.3,
                reason=f"IV Rank {ivr:.0f}, IV/HV={iv_hv_ratio:.2f}: No clear edge"
            ))

    return signals
