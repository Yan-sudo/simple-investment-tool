"""
Live Market Simulation Module
================================

Provides live (or simulated-live) market capabilities:
- Fetches real-time data from Alpaca when available, falls back to synthetic
- Runs backtests on recent market data
- Generates streaming-style time-series simulations at accelerated speed

Classes:
    LiveSimulator   -- orchestrates data fetching, backtesting, and streaming
    MarketSnapshot  -- point-in-time snapshot of a symbol's market state

Functions:
    generate_live_simulation -- convenience wrapper that produces a full
                                simulation result plus daily snapshots
"""

import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple, Generator

from data.market_data import generate_synthetic_data, MarketData
from options.backtest import OptionBacktestEngine, OptionBacktestResult
from options.strategies import (
    IronCondorStrategy,
    VerticalSpreadStrategy,
    WheelStrategy,
    StraddleStrategy,
    IVAdaptiveStrategy,
)
from options.iv_model import generate_iv_series, iv_rank, historical_volatility

try:
    from data.alpaca_client import AlpacaConfig, AlpacaClient
except ImportError:
    AlpacaConfig = None  # type: ignore[misc,assignment]
    AlpacaClient = None  # type: ignore[misc,assignment]


# ── Strategy name -> class mapping ────────────────────────────────────────────

_STRATEGY_MAP: Dict[str, type] = {
    "iron_condor": IronCondorStrategy,
    "vertical_spread": VerticalSpreadStrategy,
    "wheel": WheelStrategy,
    "straddle": StraddleStrategy,
    "iv_adaptive": IVAdaptiveStrategy,
}


def _resolve_strategy(name: str) -> "OptionStrategy":  # noqa: F821
    """Instantiate a strategy by its short name.

    Supported names (case-insensitive):
        iron_condor, vertical_spread, wheel, straddle, iv_adaptive

    Returns a default-parameter instance of the strategy.
    Raises ValueError for unknown strategy names.
    """
    key = name.lower().replace(" ", "_").replace("-", "_")
    cls = _STRATEGY_MAP.get(key)
    if cls is None:
        available = ", ".join(sorted(_STRATEGY_MAP.keys()))
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {available}"
        )
    return cls()


# ── MarketSnapshot dataclass ──────────────────────────────────────────────────

@dataclass
class MarketSnapshot:
    """Point-in-time snapshot of a symbol's market state.

    Attributes:
        symbol:        Ticker symbol (e.g. "SPY")
        price:         Last traded / closing price
        change_pct:    Intraday percentage change
        volume:        Daily volume
        timestamp:     ISO-format timestamp string
        iv_estimate:   Estimated implied volatility (annualised)
        ivr_estimate:  Estimated IV Rank (0-100)
    """
    symbol: str
    price: float
    change_pct: float
    volume: int
    timestamp: str
    iv_estimate: float
    ivr_estimate: float

    def __repr__(self) -> str:
        direction = "+" if self.change_pct >= 0 else ""
        return (
            f"MarketSnapshot({self.symbol} ${self.price:.2f} "
            f"{direction}{self.change_pct:.2f}% "
            f"vol={self.volume:,} IV={self.iv_estimate:.1%} "
            f"IVR={self.ivr_estimate:.0f} @ {self.timestamp})"
        )


# ── LiveSimulator class ──────────────────────────────────────────────────────

class LiveSimulator:
    """Orchestrates live / simulated-live market interactions.

    Tries to source data from Alpaca first; transparently falls back to
    synthetic data when Alpaca credentials are not configured or the API
    call fails.

    Parameters:
        symbol:          Ticker to track (default "SPY")
        initial_capital: Starting cash for backtests
        strategy_name:   Short name of the option strategy to use
    """

    def __init__(
        self,
        symbol: str = "SPY",
        initial_capital: float = 100_000.0,
        strategy_name: str = "iron_condor",
        **engine_kwargs,
    ) -> None:
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.strategy_name = strategy_name
        self._engine_kwargs = engine_kwargs  # Passed through to OptionBacktestEngine

        # Try to create an Alpaca client (may be unavailable)
        self._alpaca: Optional[object] = None
        if AlpacaClient is not None:
            try:
                config = AlpacaConfig.from_env()
                if config.is_configured:
                    client = AlpacaClient(config)
                    if client.is_connected:
                        self._alpaca = client
            except Exception:
                self._alpaca = None

    # ── Alpaca availability check ─────────────────────────────────────────

    @property
    def has_live_data(self) -> bool:
        """True when a working Alpaca connection is available."""
        return self._alpaca is not None

    # ── Data fetching ─────────────────────────────────────────────────────

    def fetch_current_data(self, days: int = 500, seed: int = 42) -> MarketData:
        """Return the most recent *days* of market data.

        Behaviour:
            1. If Alpaca is connected, fetch real bars and convert to MarketData.
            2. Otherwise generate synthetic data anchored to today's date so the
               time-series ends on the current date.

        Args:
            days: Number of trading days to retrieve (default 500, ~2 years).

        Returns:
            MarketData object with OHLCV data.
        """
        # Attempt Alpaca first
        if self._alpaca is not None:
            try:
                data = self._alpaca.fetch_market_data(self.symbol, days=days)
                if len(data) > 0:
                    return data
            except Exception:
                pass  # Fall through to synthetic

        # Synthetic fallback: anchor end-date to today
        return self._generate_recent_synthetic(days, seed=seed)

    # ── Backtesting ───────────────────────────────────────────────────────

    def run_live_backtest(self, strategy_or_data=None, data=None) -> OptionBacktestResult:
        """Run an option strategy on the supplied data.

        Accepts either:
            run_live_backtest(data)          — uses self.strategy_name
            run_live_backtest(strategy, data) — uses the given strategy object

        Returns:
            OptionBacktestResult with full performance metrics.
        """
        if data is None:
            # Called as run_live_backtest(data) — single arg
            data = strategy_or_data
            strategy = _resolve_strategy(self.strategy_name)
        else:
            # Called as run_live_backtest(strategy, data)
            strategy = strategy_or_data
        engine = OptionBacktestEngine(
            initial_capital=self.initial_capital, **self._engine_kwargs
        )
        return engine.run(strategy, data)

    # ── Current snapshot ──────────────────────────────────────────────────

    def get_current_snapshot(self, symbol: Optional[str] = None) -> MarketSnapshot:
        """Return a current-moment snapshot for the symbol.

        Tries Alpaca's snapshot endpoint first; on failure, builds a
        synthetic snapshot from the most recent bar of generated data.

        Args:
            symbol: Override the default symbol (optional).

        Returns:
            MarketSnapshot with price, change, volume, IV, and IVR.
        """
        sym = symbol or self.symbol
        now_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        # Attempt Alpaca snapshot
        if self._alpaca is not None:
            try:
                snap = self._alpaca.get_snapshot(sym)
                trade = snap.get("latestTrade", snap.get("latest_trade", {}))
                bar = snap.get("dailyBar", snap.get("daily_bar", {}))
                price = float(trade.get("p", trade.get("Price", 0)))
                prev_close = float(bar.get("c", bar.get("Close", price)))
                volume = int(bar.get("v", bar.get("Volume", 0)))
                change_pct = (
                    ((price - prev_close) / prev_close * 100)
                    if prev_close > 0
                    else 0.0
                )
                # Rough IV / IVR estimates from recent data
                iv_est, ivr_est = self._estimate_iv_from_alpaca(sym)
                return MarketSnapshot(
                    symbol=sym,
                    price=round(price, 2),
                    change_pct=round(change_pct, 2),
                    volume=volume,
                    timestamp=now_str,
                    iv_estimate=round(iv_est, 4),
                    ivr_estimate=round(ivr_est, 1),
                )
            except Exception:
                pass  # Fall through to synthetic

        # Synthetic snapshot
        return self._synthetic_snapshot(sym, now_str)

    # ── Real-time simulation generator ────────────────────────────────────

    def simulate_realtime(
        self,
        days: int = 30,
        speed: float = 1.0,
        seed: Optional[int] = None,
    ) -> Generator[Tuple[str, float, str, float], None, None]:
        """Yield a streaming time-series simulation of trading activity.

        Each iteration produces one simulated trading day and sleeps for
        a short interval controlled by *speed* to mimic real-time pacing.

        Args:
            days:  Number of trading days to simulate.
            speed: Speed multiplier.  1.0 = 1 second per day,
                   2.0 = 0.5 seconds per day, etc.
                   Use 0 or negative for no delay (instant).

        Yields:
            Tuples of (date_str, price, signal, portfolio_value) where:
                date_str:        "YYYY-MM-DD"
                price:           Simulated closing price
                signal:          One of "BUY_PREMIUM", "SELL_PREMIUM", "HOLD"
                portfolio_value: Running portfolio mark-to-market
        """
        rng = random.Random(seed)
        today = datetime.now()

        # We need enough data history for the strategy (252+ days warmup).
        # Generate a longer synthetic series; the last *days* entries are
        # the ones we yield.
        total_days = days + 400  # extra warmup
        data = self._generate_recent_synthetic(total_days, seed=seed)

        if len(data) == 0:
            return

        # Run the backtest to get portfolio values and signals
        strategy = _resolve_strategy(self.strategy_name)
        engine = OptionBacktestEngine(initial_capital=self.initial_capital)
        result = engine.run(strategy, data)

        # Build a set of entry dates for signal mapping
        entry_dates = set()
        for trade in result.trades:
            entry_dates.add(trade.entry_date)

        # Determine the slice to stream
        n = len(data)
        stream_start = max(0, n - days)

        # Compute IV series for signal descriptions
        iv_series = generate_iv_series(data.close, base_iv=0.20, seed=42)
        hv_series = historical_volatility(data.close, window=20)

        delay = (1.0 / speed) if speed > 0 else 0.0

        for i in range(stream_start, n):
            date_str = data.dates[i]
            price = data.close[i]

            # Determine signal from IV analysis
            if i >= 252 and i < len(iv_series) and i < len(hv_series):
                iv_hist = iv_series[max(0, i - 252):i]
                ivr_val = iv_rank(iv_series[i], iv_hist)
                iv_hv = iv_series[i] / hv_series[i] if hv_series[i] > 0 else 1.0
                if ivr_val > 60 and iv_hv > 1.1:
                    signal = "SELL_PREMIUM"
                elif ivr_val < 30 and iv_hv < 0.95:
                    signal = "BUY_PREMIUM"
                else:
                    signal = "HOLD"
            else:
                signal = "HOLD"

            portfolio_value = result.portfolio_values[i]

            yield (date_str, round(price, 2), signal, round(portfolio_value, 2))

            if delay > 0:
                time.sleep(delay)

    # ── Snapshot helpers ────────────────────────────────────────────────

    def get_snapshots(self, data: MarketData) -> List[MarketSnapshot]:
        """Build daily MarketSnapshot series from MarketData.

        Useful for plotting price + IV overlay in the GUI.
        """
        iv_series = generate_iv_series(data.close, base_iv=0.20, seed=42)
        hv = historical_volatility(data.close, window=20)
        snapshots: List[MarketSnapshot] = []
        for i in range(len(data)):
            prev_close = data.close[i - 1] if i > 0 else data.close[i]
            change = ((data.close[i] - prev_close) / prev_close * 100) if prev_close > 0 else 0
            iv = iv_series[i] if i < len(iv_series) else 0.20
            # IVR over last 60 days
            iv_window = iv_series[max(0, i - 60):i]
            ivr = iv_rank(iv, iv_window) if iv_window else 50.0
            snapshots.append(MarketSnapshot(
                symbol=self.symbol,
                price=data.close[i],
                change_pct=change,
                volume=int(data.volume[i]) if hasattr(data, 'volume') and data.volume else 0,
                timestamp=str(data.dates[i]),
                iv_estimate=iv,
                ivr_estimate=ivr,
            ))
        return snapshots

    # ── Internal helpers ──────────────────────────────────────────────────

    def _generate_recent_synthetic(
        self,
        days: int = 500,
        seed: Optional[int] = None,
    ) -> MarketData:
        """Generate synthetic data whose last date is close to today.

        The standard ``generate_synthetic_data`` starts from 2023-01-02.
        This helper shifts the date axis so the series ends on the current
        date, giving the illusion of "recent" market data.
        """
        raw = generate_synthetic_data(
            symbol=self.symbol,
            days=int(days * 1.6),  # over-generate to compensate weekends
            start_price=450.0 if self.symbol.upper() == "SPY" else 100.0,
            volatility=0.015,
            trend=0.0003,
            seed=seed,
        )

        if len(raw) == 0:
            return raw

        # Trim to requested length
        if len(raw) > days:
            raw = raw.get_slice(len(raw) - days, len(raw))

        # Shift dates so the last bar falls on today (or the last weekday)
        today = datetime.now()
        while today.weekday() >= 5:
            today -= timedelta(days=1)

        n = len(raw)
        new_dates: List[str] = []
        cursor = today - timedelta(days=(n - 1) * 2)  # overshoot back
        idx = 0
        while idx < n:
            if cursor.weekday() < 5:
                new_dates.append(cursor.strftime("%Y-%m-%d"))
                idx += 1
            cursor += timedelta(days=1)

        # Replace dates in-place
        raw.dates = new_dates[:n]
        return raw

    def _estimate_iv_from_alpaca(
        self, symbol: str
    ) -> Tuple[float, float]:
        """Estimate IV and IVR from recent Alpaca bars (rough proxy).

        Fetches 300 days of daily closes, computes 20-day HV as an IV
        proxy, and derives IV Rank from the 252-day HV history.

        Returns:
            (iv_estimate, ivr_estimate)
        """
        try:
            data = self._alpaca.fetch_market_data(symbol, days=300)
            if len(data) < 30:
                return 0.20, 50.0
            hv = historical_volatility(data.close, window=20)
            current_hv = hv[-1] if hv[-1] > 0 else 0.20
            # Use HV as a rough IV proxy (IV typically carries a premium)
            iv_est = current_hv * 1.15
            hv_history = [v for v in hv[-252:] if v > 0]
            ivr_est = iv_rank(iv_est, hv_history) if hv_history else 50.0
            return iv_est, ivr_est
        except Exception:
            return 0.20, 50.0

    def _synthetic_snapshot(
        self, symbol: str, timestamp: str
    ) -> MarketSnapshot:
        """Build a MarketSnapshot from synthetic data.

        Generates a short synthetic series and derives the snapshot from
        the last two bars.
        """
        rng = random.Random()
        data = generate_synthetic_data(
            symbol=symbol,
            days=60,
            start_price=450.0 if symbol.upper() == "SPY" else 100.0,
            volatility=0.015,
            trend=0.0003,
        )

        if len(data) < 2:
            return MarketSnapshot(
                symbol=symbol,
                price=100.0,
                change_pct=0.0,
                volume=1_000_000,
                timestamp=timestamp,
                iv_estimate=0.20,
                ivr_estimate=50.0,
            )

        price = data.close[-1]
        prev = data.close[-2]
        change_pct = ((price - prev) / prev * 100) if prev > 0 else 0.0
        volume = data.volume[-1]

        # Quick IV / IVR from the short series
        iv_series = generate_iv_series(data.close, base_iv=0.20, seed=42)
        current_iv = iv_series[-1]
        iv_hist = iv_series[:-1] if len(iv_series) > 1 else iv_series
        ivr_val = iv_rank(current_iv, iv_hist)

        return MarketSnapshot(
            symbol=symbol,
            price=round(price, 2),
            change_pct=round(change_pct, 2),
            volume=volume,
            timestamp=timestamp,
            iv_estimate=round(current_iv, 4),
            ivr_estimate=round(ivr_val, 1),
        )


# ── Convenience function ──────────────────────────────────────────────────────

def generate_live_simulation(
    symbol: str = "SPY",
    days: int = 500,
    strategy_name: str = "iron_condor",
    initial_capital: float = 100_000.0,
    seed: Optional[int] = 42,
) -> Tuple[OptionBacktestResult, List[MarketSnapshot]]:
    """Run a full simulation and return the result plus daily snapshots.

    This is the main convenience entry-point.  It:
        1. Generates synthetic data anchored to the recent past.
        2. Runs the named strategy via the backtest engine.
        3. Builds a list of MarketSnapshot objects for every simulated day.

    Args:
        symbol:          Ticker symbol (default "SPY").
        days:            Number of calendar days to generate (trading days
                         will be fewer due to weekends).
        strategy_name:   One of the supported strategy short names.
        initial_capital: Starting portfolio cash.
        seed:            Random seed for reproducibility.

    Returns:
        A 2-tuple of (OptionBacktestResult, List[MarketSnapshot]).
    """
    sim = LiveSimulator(
        symbol=symbol,
        initial_capital=initial_capital,
        strategy_name=strategy_name,
    )

    # Generate recent synthetic data
    data = sim._generate_recent_synthetic(days=days, seed=seed)

    if len(data) == 0:
        raise RuntimeError(
            f"Failed to generate synthetic data for {symbol} ({days} days)"
        )

    # Run backtest
    result = sim.run_live_backtest(data)

    # Build daily snapshots
    iv_series = generate_iv_series(data.close, base_iv=0.20, seed=42)
    hv_series = historical_volatility(data.close, window=20)

    snapshots: List[MarketSnapshot] = []
    for i in range(len(data)):
        price = data.close[i]
        prev_price = data.close[i - 1] if i > 0 else price
        change_pct = ((price - prev_price) / prev_price * 100) if prev_price > 0 else 0.0
        volume = data.volume[i]

        current_iv = iv_series[i] if i < len(iv_series) else 0.20
        iv_hist = iv_series[max(0, i - 252):i] if i > 0 else [current_iv]
        ivr_val = iv_rank(current_iv, iv_hist) if iv_hist else 50.0

        snapshots.append(MarketSnapshot(
            symbol=symbol,
            price=round(price, 2),
            change_pct=round(change_pct, 2),
            volume=volume,
            timestamp=data.dates[i],
            iv_estimate=round(current_iv, 4),
            ivr_estimate=round(ivr_val, 1),
        ))

    return result, snapshots
