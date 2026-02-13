"""
News / Market Sentiment Analysis Module

Detects "news-like events" from price action and uses them as sentiment signals
for the option trading system. In backtesting mode, real news is unavailable, so
this module infers sentiment from gaps, volatility spikes/crushes, trend breaks,
and momentum shifts. In live mode (Alpaca configured), it can supplement with
real headline data.

Signal flow:
    Price data -> MarketSentimentAnalyzer.analyze() -> List[SentimentSignal]
    SentimentSignal -> SentimentFilter.should_trade() -> bool
    SentimentSignal -> SentimentFilter.adjust_position_size() -> float
"""

import math
from dataclasses import dataclass
from typing import List, Optional


# ── Sentiment Signal Dataclass ───────────────────────────────────────────────

@dataclass
class SentimentSignal:
    """A single day's sentiment reading derived from price action."""

    date: str
    score: float            # -1.0 (extremely bearish) to +1.0 (extremely bullish)
    magnitude: float        # 0.0 to 1.0, how strong the signal is
    event_type: str         # gap_down, gap_up, vol_spike, vol_crush,
                            # trend_break, momentum_shift, range_bound, normal
    description: str        # Human-readable event description
    actionable: bool        # Whether this should affect trading decisions

    def __repr__(self):
        arrow = "+" if self.score >= 0 else ""
        tag = " [ACTIONABLE]" if self.actionable else ""
        return (
            f"SentimentSignal({self.date}: {arrow}{self.score:.2f} "
            f"mag={self.magnitude:.2f} {self.event_type}{tag})"
        )


# ── Helper Functions ─────────────────────────────────────────────────────────

def _sma(values: List[float], window: int, idx: int) -> float:
    """Simple moving average ending at idx (inclusive), using *window* bars."""
    if idx < window - 1:
        return 0.0
    segment = values[idx - window + 1: idx + 1]
    return sum(segment) / len(segment)


def _atr(highs: List[float], lows: List[float], closes: List[float],
         window: int, idx: int) -> float:
    """Average True Range ending at idx, using *window* bars."""
    if idx < window:
        return 0.0
    tr_sum = 0.0
    for j in range(idx - window + 1, idx + 1):
        hl = highs[j] - lows[j]
        if j > 0:
            hc = abs(highs[j] - closes[j - 1])
            lc = abs(lows[j] - closes[j - 1])
            tr_sum += max(hl, hc, lc)
        else:
            tr_sum += hl
    return tr_sum / window


def _rsi(closes: List[float], period: int, idx: int) -> float:
    """Relative Strength Index at idx using *period* lookback.

    Returns 50.0 when insufficient data.
    """
    if idx < period:
        return 50.0

    gains = 0.0
    losses = 0.0
    for j in range(idx - period + 1, idx + 1):
        change = closes[j] - closes[j - 1]
        if change > 0:
            gains += change
        else:
            losses += abs(change)

    avg_gain = gains / period
    avg_loss = losses / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


# ── Market Sentiment Analyzer ────────────────────────────────────────────────

class MarketSentimentAnalyzer:
    """Detect news-like events from price action and generate sentiment signals.

    This replaces actual news feeds during backtesting. Each day is classified
    into an event type (gap, vol spike, trend break, etc.) and scored from
    -1.0 (bearish) to +1.0 (bullish). When multiple signals fire on the same
    day, the one with the highest magnitude wins.

    Parameters:
        gap_threshold:   Minimum close-to-close return to consider a gap (default 1.5%)
        vol_spike_mult:  Multiple of 20-day average range to flag a vol spike (default 1.5x)
        trend_window:    SMA period for trend-break detection (default 20)
        momentum_window: RSI period for momentum-shift detection (default 10)
    """

    def __init__(
        self,
        gap_threshold: float = 0.015,
        vol_spike_mult: float = 1.5,
        trend_window: int = 20,
        momentum_window: int = 10,
    ):
        self.gap_threshold = gap_threshold
        self.vol_spike_mult = vol_spike_mult
        self.trend_window = trend_window
        self.momentum_window = momentum_window

    # ── Main entry point ─────────────────────────────────────────────────

    def analyze(
        self,
        closes: List[float],
        highs: List[float],
        lows: List[float],
        volumes: Optional[List[float]] = None,
        iv_series: Optional[List[float]] = None,
        dates: Optional[List[str]] = None,
    ) -> List[SentimentSignal]:
        """Generate one SentimentSignal per day from price action.

        Args:
            closes:    Daily closing prices.
            highs:     Daily high prices.
            lows:      Daily low prices.
            volumes:   Daily volumes (optional, reserved for future use).
            iv_series: Daily implied-volatility values (optional, enables vol-crush detection).
            dates:     Date strings (optional; day indices used if omitted).

        Returns:
            List of SentimentSignal, one per bar.
        """
        n = len(closes)
        if n == 0:
            return []

        if dates is None:
            dates = [str(i) for i in range(n)]

        signals: List[SentimentSignal] = []

        for i in range(n):
            candidates: List[SentimentSignal] = []

            # ── a. Gap detection ─────────────────────────────────────────
            if i >= 1:
                gap_signal = self._detect_gap(closes, highs, lows, dates, i)
                if gap_signal is not None:
                    candidates.append(gap_signal)

            # ── b. Volatility spike ──────────────────────────────────────
            if i >= 20:
                vol_signal = self._detect_vol_spike(closes, highs, lows, dates, i)
                if vol_signal is not None:
                    candidates.append(vol_signal)

            # ── c. Vol crush (IV-based) ──────────────────────────────────
            if iv_series is not None and i >= 3:
                crush_signal = self._detect_vol_crush(iv_series, dates, i)
                if crush_signal is not None:
                    candidates.append(crush_signal)

            # ── d. Trend break ───────────────────────────────────────────
            if i >= self.trend_window:
                trend_signal = self._detect_trend_break(
                    closes, highs, lows, dates, i
                )
                if trend_signal is not None:
                    candidates.append(trend_signal)

            # ── e. Momentum shift ────────────────────────────────────────
            if i >= self.momentum_window + 1:
                mom_signal = self._detect_momentum_shift(closes, dates, i)
                if mom_signal is not None:
                    candidates.append(mom_signal)

            # ── Pick the strongest signal, or default to normal ──────────
            if candidates:
                best = max(candidates, key=lambda s: s.magnitude)
                signals.append(best)
            else:
                signals.append(self._normal_signal(dates, i))

        return signals

    # ── Private detection helpers ─────────────────────────────────────────

    def _detect_gap(
        self,
        closes: List[float],
        highs: List[float],
        lows: List[float],
        dates: List[str],
        i: int,
    ) -> Optional[SentimentSignal]:
        """Detect gap-up or gap-down from close[i-1] to close[i].

        A "true" gap requires the intraday range not to bridge back:
          - Gap up:   low[i] > close[i-1]
          - Gap down: high[i] < close[i-1]
        """
        prev_close = closes[i - 1]
        if prev_close == 0:
            return None
        ret = (closes[i] - prev_close) / prev_close

        if abs(ret) < self.gap_threshold:
            return None

        if ret > 0 and lows[i] > prev_close:
            # Gap up confirmed
            magnitude = min(1.0, abs(ret) / 0.05)  # 5% move = mag 1.0
            score = 0.7
            return SentimentSignal(
                date=dates[i],
                score=score,
                magnitude=magnitude,
                event_type="gap_up",
                description=(
                    f"Gap up {ret:+.1%}: price opened above prior close "
                    f"({prev_close:.2f}) with low {lows[i]:.2f} holding the gap"
                ),
                actionable=True,
            )

        if ret < 0 and highs[i] < prev_close:
            # Gap down confirmed
            magnitude = min(1.0, abs(ret) / 0.05)
            score = -0.7
            return SentimentSignal(
                date=dates[i],
                score=score,
                magnitude=magnitude,
                event_type="gap_down",
                description=(
                    f"Gap down {ret:+.1%}: price opened below prior close "
                    f"({prev_close:.2f}) with high {highs[i]:.2f} failing to fill"
                ),
                actionable=True,
            )

        return None

    def _detect_vol_spike(
        self,
        closes: List[float],
        highs: List[float],
        lows: List[float],
        dates: List[str],
        i: int,
    ) -> Optional[SentimentSignal]:
        """Detect an abnormally wide intraday range relative to the 20-day average."""
        if closes[i] == 0:
            return None

        today_range = (highs[i] - lows[i]) / closes[i]

        # 20-day average range
        avg_range = 0.0
        count = 0
        for j in range(i - 20, i):
            if closes[j] > 0:
                avg_range += (highs[j] - lows[j]) / closes[j]
                count += 1
        if count == 0:
            return None
        avg_range /= count

        if avg_range == 0 or today_range <= self.vol_spike_mult * avg_range:
            return None

        # Determine direction: close near low = bearish, close near high = bullish
        hl_range = highs[i] - lows[i]
        if hl_range == 0:
            return None
        close_position = (closes[i] - lows[i]) / hl_range  # 0 = at low, 1 = at high

        magnitude = min(1.0, today_range / (3.0 * avg_range))

        if close_position < 0.4:
            # Close near the low -- bearish vol spike
            score = -0.5
            desc = (
                f"Bearish vol spike: range {today_range:.2%} is "
                f"{today_range / avg_range:.1f}x the 20-day avg, "
                f"close near low ({close_position:.0%} of range)"
            )
        else:
            # Close near the high (or middle) -- bullish vol spike
            score = 0.3
            desc = (
                f"Bullish vol spike: range {today_range:.2%} is "
                f"{today_range / avg_range:.1f}x the 20-day avg, "
                f"close near high ({close_position:.0%} of range)"
            )

        return SentimentSignal(
            date=dates[i],
            score=score,
            magnitude=magnitude,
            event_type="vol_spike",
            description=desc,
            actionable=abs(score) >= 0.3,
        )

    def _detect_vol_crush(
        self,
        iv_series: List[float],
        dates: List[str],
        i: int,
    ) -> Optional[SentimentSignal]:
        """Detect IV crush: IV drops > 15% over the last 3 days."""
        iv_now = iv_series[i]
        iv_3d_ago = iv_series[i - 3]
        if iv_3d_ago == 0:
            return None

        iv_change = (iv_now - iv_3d_ago) / iv_3d_ago

        if iv_change >= -0.15:
            return None

        magnitude = min(1.0, abs(iv_change) / 0.30)
        score = 0.2
        return SentimentSignal(
            date=dates[i],
            score=score,
            magnitude=magnitude,
            event_type="vol_crush",
            description=(
                f"Vol crush: IV fell {iv_change:.1%} over 3 days "
                f"({iv_3d_ago:.1%} -> {iv_now:.1%}), fear subsiding"
            ),
            actionable=False,  # score 0.2 < 0.3
        )

    def _detect_trend_break(
        self,
        closes: List[float],
        highs: List[float],
        lows: List[float],
        dates: List[str],
        i: int,
    ) -> Optional[SentimentSignal]:
        """Detect price crossing above/below the trend SMA by more than 1 ATR."""
        sma_val = _sma(closes, self.trend_window, i)
        if sma_val == 0:
            return None

        atr_val = _atr(highs, lows, closes, self.trend_window, i)
        if atr_val == 0:
            return None

        deviation = closes[i] - sma_val

        if deviation > atr_val:
            # Break above
            magnitude = min(1.0, deviation / (2.0 * atr_val))
            score = 0.4
            return SentimentSignal(
                date=dates[i],
                score=score,
                magnitude=magnitude,
                event_type="trend_break",
                description=(
                    f"Bullish trend break: close {closes[i]:.2f} is "
                    f"{deviation / atr_val:.1f} ATRs above {self.trend_window}-day "
                    f"SMA ({sma_val:.2f})"
                ),
                actionable=True,
            )

        if deviation < -atr_val:
            # Break below
            magnitude = min(1.0, abs(deviation) / (2.0 * atr_val))
            score = -0.4
            return SentimentSignal(
                date=dates[i],
                score=score,
                magnitude=magnitude,
                event_type="trend_break",
                description=(
                    f"Bearish trend break: close {closes[i]:.2f} is "
                    f"{abs(deviation) / atr_val:.1f} ATRs below {self.trend_window}-day "
                    f"SMA ({sma_val:.2f})"
                ),
                actionable=True,
            )

        return None

    def _detect_momentum_shift(
        self,
        closes: List[float],
        dates: List[str],
        i: int,
    ) -> Optional[SentimentSignal]:
        """Detect RSI crossing from oversold/overbought territory back to neutral."""
        rsi_today = _rsi(closes, self.momentum_window, i)
        rsi_yesterday = _rsi(closes, self.momentum_window, i - 1)

        # Oversold reversal: RSI was below 30 yesterday, now above 30
        if rsi_yesterday < 30.0 and rsi_today >= 30.0:
            magnitude = min(1.0, (30.0 - rsi_yesterday) / 20.0)
            return SentimentSignal(
                date=dates[i],
                score=0.3,
                magnitude=magnitude,
                event_type="momentum_shift",
                description=(
                    f"Oversold reversal: RSI({self.momentum_window}) crossed up "
                    f"through 30 ({rsi_yesterday:.1f} -> {rsi_today:.1f})"
                ),
                actionable=True,
            )

        # Overbought reversal: RSI was above 70 yesterday, now below 70
        if rsi_yesterday > 70.0 and rsi_today <= 70.0:
            magnitude = min(1.0, (rsi_yesterday - 70.0) / 20.0)
            return SentimentSignal(
                date=dates[i],
                score=-0.3,
                magnitude=magnitude,
                event_type="momentum_shift",
                description=(
                    f"Overbought reversal: RSI({self.momentum_window}) crossed down "
                    f"through 70 ({rsi_yesterday:.1f} -> {rsi_today:.1f})"
                ),
                actionable=True,
            )

        return None

    def _normal_signal(self, dates: List[str], i: int) -> SentimentSignal:
        """Return a neutral / normal-day signal."""
        return SentimentSignal(
            date=dates[i],
            score=0.0,
            magnitude=0.0,
            event_type="normal",
            description="Normal trading day, no significant price-action events",
            actionable=False,
        )


# ── Sentiment Filter ─────────────────────────────────────────────────────────

class SentimentFilter:
    """Gate and size trades based on sentiment signals.

    For long-only option strategies (e.g. credit put spreads, covered calls),
    this filter blocks entries during extreme fear and scales position size
    based on sentiment alignment.

    Parameters:
        min_score:             Lowest accepted score (default -1.0, no floor)
        max_score:             Highest accepted score (default +1.0, no ceiling)
        block_on_extreme_fear: Whether to block long entries during extreme bearish events
        fear_threshold:        Score below which fear is considered extreme (default -0.6)
    """

    def __init__(
        self,
        min_score: float = -1.0,
        max_score: float = 1.0,
        block_on_extreme_fear: bool = True,
        fear_threshold: float = -0.6,
    ):
        self.min_score = min_score
        self.max_score = max_score
        self.block_on_extreme_fear = block_on_extreme_fear
        self.fear_threshold = fear_threshold

    def should_trade(
        self, signal: SentimentSignal, direction: str = "long"
    ) -> bool:
        """Decide whether to allow a trade given the current sentiment.

        Rules for long-only trading:
          1. Block if score < fear_threshold and block_on_extreme_fear is on.
          2. Allow if signal is not actionable (neutral day -- no reason to block).
          3. Allow if signal direction matches trade direction.
          4. Block bearish actionable signals when going long (score < -0.3).

        Args:
            signal:    Today's SentimentSignal.
            direction: Trade direction, "long" or "short".

        Returns:
            True if the trade should proceed, False if blocked.
        """
        # Rule 1: extreme fear block
        if (
            direction == "long"
            and self.block_on_extreme_fear
            and signal.score < self.fear_threshold
        ):
            return False

        # Rule 2: non-actionable signals never block
        if not signal.actionable:
            return True

        # Rule 3 & 4: directional alignment
        if direction == "long":
            # Allow bullish or neutral-ish actionable signals
            if signal.score >= -0.3:
                return True
            # Block bearish actionable signals for long entries
            return False

        if direction == "short":
            # Mirror logic: block bullish actionable signals for shorts
            if signal.score <= 0.3:
                return True
            return False

        # Unknown direction -- default allow
        return True

    def adjust_position_size(
        self, signal: SentimentSignal, base_size: float = 1.0
    ) -> float:
        """Scale position size based on how well sentiment aligns with a long trade.

        Scaling table (long bias):
            Very bullish   (score > +0.5)    -> 1.25x
            Mildly bullish (+0.2 to +0.5)    -> 1.10x
            Neutral        (-0.2 to +0.2)    -> 1.00x
            Mildly bearish (-0.5 to -0.2)    -> 0.75x
            Very bearish   (score < -0.5)    -> 0.50x

        Args:
            signal:    Today's SentimentSignal.
            base_size: Baseline position size (default 1.0 = 100%).

        Returns:
            Adjusted position size.
        """
        score = signal.score

        if score > 0.5:
            multiplier = 1.25
        elif score > 0.2:
            multiplier = 1.10
        elif score >= -0.2:
            multiplier = 1.00
        elif score >= -0.5:
            multiplier = 0.75
        else:
            multiplier = 0.50

        return base_size * multiplier


# ── Report Formatting ────────────────────────────────────────────────────────

def format_sentiment_report(
    signals: List[SentimentSignal], last_n: int = 10
) -> str:
    """Format a human-readable summary of recent sentiment signals.

    Args:
        signals: Full list of SentimentSignal objects.
        last_n:  Number of most-recent signals to display (default 10).

    Returns:
        Formatted multi-line string.
    """
    if not signals:
        return "No sentiment signals available."

    recent = signals[-last_n:]

    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("  MARKET SENTIMENT REPORT")
    lines.append("=" * 72)
    lines.append(
        f"  Showing last {len(recent)} of {len(signals)} total signals"
    )
    lines.append("-" * 72)
    lines.append(
        f"  {'Date':<12} {'Score':>6} {'Mag':>5} {'Type':<17} {'Act':>3}  Description"
    )
    lines.append("-" * 72)

    for sig in recent:
        act_marker = " * " if sig.actionable else "   "
        score_str = f"{sig.score:+.2f}"
        mag_str = f"{sig.magnitude:.2f}"
        # Truncate description for table width
        desc = sig.description
        if len(desc) > 45:
            desc = desc[:42] + "..."
        lines.append(
            f"  {sig.date:<12} {score_str:>6} {mag_str:>5} {sig.event_type:<17}"
            f"{act_marker}  {desc}"
        )

    lines.append("-" * 72)

    # Summary statistics
    actionable_signals = [s for s in recent if s.actionable]
    avg_score = sum(s.score for s in recent) / len(recent) if recent else 0.0
    bullish_count = sum(1 for s in recent if s.score > 0.2)
    bearish_count = sum(1 for s in recent if s.score < -0.2)
    neutral_count = len(recent) - bullish_count - bearish_count

    lines.append(f"  Avg score:  {avg_score:+.3f}")
    lines.append(
        f"  Bullish: {bullish_count}  |  Bearish: {bearish_count}  |  "
        f"Neutral: {neutral_count}"
    )
    lines.append(f"  Actionable signals: {len(actionable_signals)} of {len(recent)}")

    if avg_score > 0.2:
        bias = "BULLISH"
    elif avg_score < -0.2:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"
    lines.append(f"  Overall bias: {bias}")
    lines.append("=" * 72)

    return "\n".join(lines)
