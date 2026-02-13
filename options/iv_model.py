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
from dataclasses import dataclass, field


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
                       iv_mean_reversion: float = 0.035,
                       iv_vol_of_vol: float = 0.15,
                       leverage_effect: float = -0.5,
                       iv_premium: float = 0.03,
                       seed: Optional[int] = None) -> List[float]:
    """Generate a synthetic IV time series that behaves realistically.

    Key properties:
    - Mean-reverting around base_iv
    - Negatively correlated with returns (leverage effect)
    - Has its own stochastic component (vol-of-vol)
    - Clusters (high vol tends to stay high)
    - IV carries a variance risk premium over HV (typically 15-20%)

    Args:
        closes: Underlying closing prices
        base_iv: Long-term mean IV
        iv_mean_reversion: Speed of mean reversion (higher = faster)
        iv_vol_of_vol: Volatility of volatility
        leverage_effect: Correlation between returns and IV changes
        iv_premium: Variance risk premium — IV floats above HV by this amount
        seed: Random seed

    Returns:
        List of IV values, one per day
    """
    import random as rng
    if seed is not None:
        rng.seed(seed)

    n = len(closes)
    target_iv = base_iv + iv_premium  # IV trades at a premium to realised vol
    iv_series = [target_iv] * n

    for i in range(1, n):
        prev_iv = iv_series[i - 1]

        # Mean reversion pull toward target (includes premium)
        # iv_mean_reversion is daily fraction (0.035 ≈ half-life of ~20 days)
        mr_pull = iv_mean_reversion * (target_iv - prev_iv)

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


# ── Put/Call Ratio (PCR) Signal ─────────────────────────────────────────────
# Inspired by: PyPatel/Options-Trading-Strategies-in-Python (PCR_strategy.py)
# PCR = total put volume / total call volume
# - High PCR (>1.2): heavy put buying = fear = contrarian BULLISH
# - Low PCR (<0.5): heavy call buying = complacency = contrarian BEARISH
# Entry: PCR crosses Bollinger Band boundary
# Exit: PCR reverts to 20-day moving average

@dataclass
class PCRSignal:
    """Signal generated from Put/Call Ratio Bollinger Band analysis."""
    signal: int          # 1 = contrarian bullish, -1 = contrarian bearish, 0 = neutral
    pcr_current: float
    pcr_mean: float
    pcr_upper: float     # Bollinger upper band
    pcr_lower: float     # Bollinger lower band
    reason: str

    def __repr__(self):
        labels = {1: "CONTRARIAN_BULL", -1: "CONTRARIAN_BEAR", 0: "NEUTRAL"}
        return (f"PCRSignal({labels[self.signal]}: "
                f"PCR={self.pcr_current:.2f} vs bands "
                f"[{self.pcr_lower:.2f}, {self.pcr_upper:.2f}])")


def generate_pcr_series(closes: List[float], iv_series: List[float],
                        base_pcr: float = 0.70,
                        seed: Optional[int] = None) -> List[float]:
    """Generate a synthetic Put/Call Ratio time series.

    PCR properties:
    - Mean-reverting around base_pcr (~0.7 for equity markets)
    - Positively correlated with IV (more fear → more put buying)
    - Negatively correlated with daily returns (sell-off → PCR spikes)
    - Range: 0.3 – 2.5

    Args:
        closes: Underlying price history
        iv_series: Implied volatility series (for correlation)
        base_pcr: Long-run mean PCR (equity markets ~0.70)
        seed: Random seed

    Returns:
        List of daily PCR values
    """
    import random as rng
    if seed is not None:
        rng.seed(seed + 77)  # offset from iv_series seed

    n = len(closes)
    pcr = [base_pcr] * n
    pcr_vol = 0.06  # daily noise

    for i in range(1, n):
        prev = pcr[i - 1]

        # Mean reversion pull (slow)
        mr = 0.06 * (base_pcr - prev) / 252.0 * 100

        # IV coupling: elevated IV → more put buying → higher PCR
        iv_now = iv_series[i] if i < len(iv_series) else 0.20
        iv_effect = (iv_now - 0.20) * 0.8  # PCR rises ~0.008 per 1% IV spike

        # Return coupling: negative returns spike PCR
        ret = 0.0
        if closes[i - 1] > 0:
            ret = (closes[i] - closes[i - 1]) / closes[i - 1]
        return_effect = -ret * 4.0  # amplified leverage

        shock = rng.gauss(0, pcr_vol / math.sqrt(252))

        new_pcr = prev + mr + iv_effect * 0.01 + return_effect + shock
        pcr[i] = max(0.30, min(new_pcr, 2.50))

    return pcr


def generate_pcr_signals(pcr_series: List[float], window: int = 20,
                         num_std: float = 2.0) -> List[PCRSignal]:
    """Generate contrarian options signals from PCR Bollinger Bands.

    Strategy (adapted from PyPatel PCR_strategy.py):
      - PCR > upper_band → extreme fear → contrarian BULLISH → sell puts (good IV)
      - PCR < lower_band → complacency → contrarian BEARISH → sell calls (good IV)
      - PCR reverts to mean → exit signal / close position

    For options context:
      signal=+1 (high PCR): Good time to sell put spreads or cash-secured puts
      signal=-1 (low PCR):  Good time to sell call spreads or iron condors
      signal=0:             No PCR-based edge

    Args:
        pcr_series: Daily PCR values
        window: Bollinger Band window (default 20 days)
        num_std: Number of standard deviations for bands (default 2.0)

    Returns:
        List of PCRSignal, one per day
    """
    n = len(pcr_series)
    signals = []

    for i in range(n):
        if i < window:
            signals.append(PCRSignal(
                signal=0, pcr_current=pcr_series[i],
                pcr_mean=0.70, pcr_upper=1.0, pcr_lower=0.40,
                reason="Insufficient data"
            ))
            continue

        window_data = pcr_series[i - window:i]
        mean = sum(window_data) / len(window_data)
        std = math.sqrt(
            sum((x - mean) ** 2 for x in window_data) / max(len(window_data) - 1, 1)
        )
        upper = mean + num_std * std
        lower = max(0.20, mean - num_std * std)
        current = pcr_series[i]

        if current > upper:
            signals.append(PCRSignal(
                signal=1, pcr_current=current,
                pcr_mean=mean, pcr_upper=upper, pcr_lower=lower,
                reason=f"PCR {current:.2f} above upper band {upper:.2f}: extreme fear → contrarian bull"
            ))
        elif current < lower:
            signals.append(PCRSignal(
                signal=-1, pcr_current=current,
                pcr_mean=mean, pcr_upper=upper, pcr_lower=lower,
                reason=f"PCR {current:.2f} below lower band {lower:.2f}: complacency → contrarian bear"
            ))
        else:
            signals.append(PCRSignal(
                signal=0, pcr_current=current,
                pcr_mean=mean, pcr_upper=upper, pcr_lower=lower,
                reason=f"PCR {current:.2f} within bands [{lower:.2f}, {upper:.2f}]: neutral"
            ))

    return signals


def generate_vix_bands(iv_series: List[float], window: int = 20,
                       num_std: float = 1.5) -> List[Tuple[float, float, float]]:
    """Compute Bollinger Bands on the IV series (VIX-style band analysis).

    Strategy (adapted from PyPatel VIX_Strategy.py):
      - IV crosses above upper band → vol spike → sell premium aggressively
      - IV crosses below lower band → vol crush → buy volatility (straddles)
      - IV reverts to mean → reduce premium-selling, look to close shorts

    Args:
        iv_series: Daily implied volatility values
        window: Rolling window for mean/std (default 20)
        num_std: Multiplier for band width (default 1.5 σ)

    Returns:
        List of (mean_iv, upper_band, lower_band) tuples, one per day
    """
    n = len(iv_series)
    bands: List[Tuple[float, float, float]] = []

    for i in range(n):
        if i < window:
            v = iv_series[i]
            bands.append((v, v * 1.25, max(0.05, v * 0.75)))
            continue

        window_data = iv_series[i - window:i]
        mean = sum(window_data) / len(window_data)
        std = math.sqrt(
            sum((x - mean) ** 2 for x in window_data) / max(len(window_data) - 1, 1)
        )
        upper = mean + num_std * std
        lower = max(0.05, mean - num_std * std)
        bands.append((mean, upper, lower))

    return bands
