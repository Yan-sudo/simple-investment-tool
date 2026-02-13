"""
Trade Reflection & Self-Correction Journal

Analyzes completed option trades to detect patterns, streaks, and regime-
specific performance.  Produces actionable adjustments (position sizing,
entry tightening, cooldown, pause) that the trading system can consume
programmatically.

Key reflection rules:
    3+ consecutive losses  -> tighten entries, reduce size to 0.75x, +2 cooldown
    5+ consecutive losses  -> pause trading (should_pause = True)
    Post-pause             -> cautious 0.5x sizing for first 3 trades
    >70% win rate (last 10)-> allow 1.25x sizing
    Per-regime tracking    -> flag underperforming vol regimes

Pattern tags detected:
    "hold_too_long"            - DTE_EXIT losses dominate
    "iv_mean_reversion_risk"   - high IV at entry, IV dropped, still lost
    "cut_losers_faster"        - avg hold of losses > avg hold of wins
    "directional_risk"         - underlying moved significantly against position
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


# ── Data Structures ────────────────────────────────────────────────────────

@dataclass
class ReflectionInsight:
    """Actionable output of the trade-journal reflection process."""

    # Streak info
    consecutive_losses: int = 0
    consecutive_wins: int = 0

    # Recent performance
    recent_win_rate: float = 0.0           # Win rate over last N trades

    # Trend
    avg_pnl_trend: str = "stable"          # "improving", "stable", "deteriorating"

    # Confidence & sizing
    confidence_score: float = 0.5          # 0.0 .. 1.0
    position_size_mult: float = 1.0        # 0.5x .. 1.5x

    # Cooldown / pause
    cooldown_adjustment: int = 0           # Extra cooldown days to add
    should_pause: bool = False             # True -> stop trading temporarily

    # Entry criteria
    entry_tightening: float = 0.0          # Additive IVR threshold bump

    # Explanations
    reasoning: List[str] = field(default_factory=list)
    lesson_tags: List[str] = field(default_factory=list)


# ── Regime Performance Snapshot ────────────────────────────────────────────

@dataclass
class RegimePerformance:
    """Performance breakdown for a single volatility regime."""
    regime: str
    trade_count: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    win_rate: float = 0.0
    avg_holding_days: float = 0.0


# ── Trade Journal ──────────────────────────────────────────────────────────

class TradeJournal:
    """Stores completed trades and produces reflection-based adjustments.

    Usage::

        journal = TradeJournal()
        for record in completed_trades:
            journal.record_trade(record)

        insight = journal.reflect()
        report  = format_reflection_report(insight)
    """

    def __init__(self, lookback: int = 10,
                 iv_drop_threshold: float = 0.03,
                 directional_move_pct: float = 0.05,
                 post_pause_cautious_count: int = 3):
        """
        Args:
            lookback: Number of recent trades used for win-rate and trend
                      calculations.
            iv_drop_threshold: Minimum absolute IV drop (entry - exit proxy)
                               to flag IV mean-reversion risk.
            directional_move_pct: Minimum |underlying move %| to tag
                                  directional risk on a losing trade.
            post_pause_cautious_count: How many trades after a pause should
                                       use cautious (0.5x) sizing.
        """
        self.lookback = lookback
        self.iv_drop_threshold = iv_drop_threshold
        self.directional_move_pct = directional_move_pct
        self.post_pause_cautious_count = post_pause_cautious_count

        self._trades: List = []  # List[OptionTradeRecord]

        # Pause tracking
        self._pause_active: bool = False
        self._trades_since_pause: int = 0

    # ── Recording ──────────────────────────────────────────────────────

    def record_trade(self, trade_record) -> None:
        """Add a completed OptionTradeRecord to the journal."""
        self._trades.append(trade_record)

        # If we are in the post-pause cautious window, tick the counter
        if self._pause_active:
            self._trades_since_pause += 1
            if self._trades_since_pause >= self.post_pause_cautious_count:
                self._pause_active = False
                self._trades_since_pause = 0

    # ── Streak helpers ─────────────────────────────────────────────────

    def get_streak(self) -> int:
        """Return the current streak count.

        Positive = consecutive wins, negative = consecutive losses.
        Zero if no trades recorded.
        """
        if not self._trades:
            return 0

        streak = 0
        # Walk backwards from most recent trade
        is_winning = self._trades[-1].exit_pnl > 0
        for trade in reversed(self._trades):
            if (trade.exit_pnl > 0) == is_winning:
                streak += 1
            else:
                break

        return streak if is_winning else -streak

    # ── Regime performance ─────────────────────────────────────────────

    def get_regime_performance(self, regime: str) -> RegimePerformance:
        """Return performance breakdown for a given vol regime.

        The regime is classified from ``iv_rank_at_entry``:
            high   : IVR >= 60
            medium : 30 <= IVR < 60
            low    : IVR < 30

        Args:
            regime: One of "high", "medium", "low".

        Returns:
            RegimePerformance dataclass.
        """
        perf = RegimePerformance(regime=regime)

        for t in self._trades:
            r = _classify_regime(t.iv_rank_at_entry)
            if r != regime:
                continue

            perf.trade_count += 1
            perf.total_pnl += t.exit_pnl
            if t.exit_pnl > 0:
                perf.wins += 1
            else:
                perf.losses += 1

        if perf.trade_count > 0:
            perf.avg_pnl = perf.total_pnl / perf.trade_count
            perf.win_rate = perf.wins / perf.trade_count
            holding_sum = sum(
                t.holding_days for t in self._trades
                if _classify_regime(t.iv_rank_at_entry) == regime
            )
            perf.avg_holding_days = holding_sum / perf.trade_count

        return perf

    # ── Core reflection ────────────────────────────────────────────────

    def reflect(self) -> ReflectionInsight:
        """Analyze recent trades and return actionable adjustments.

        Returns:
            ReflectionInsight populated with sizing, cooldown, pause,
            entry-tightening recommendations, and lesson tags.
        """
        insight = ReflectionInsight()

        if not self._trades:
            insight.reasoning.append("No trades recorded yet.")
            return insight

        # -- Streak computation --
        streak = self.get_streak()
        if streak < 0:
            insight.consecutive_losses = abs(streak)
        else:
            insight.consecutive_wins = streak

        # -- Recent window --
        recent = self._trades[-self.lookback:]
        recent_wins = [t for t in recent if t.exit_pnl > 0]
        insight.recent_win_rate = len(recent_wins) / len(recent)

        # -- PnL trend (last 5 vs prior 5) --
        insight.avg_pnl_trend = self._compute_pnl_trend()

        # -- Lesson tags (pattern detection) --
        self._detect_patterns(insight, recent)

        # -- Regime flags --
        self._flag_underperforming_regimes(insight)

        # -- Apply rules in priority order --

        # Rule: 5+ consecutive losses -> pause
        if insight.consecutive_losses >= 5:
            insight.should_pause = True
            insight.position_size_mult = 0.0
            insight.confidence_score = 0.0
            insight.cooldown_adjustment = 10
            insight.entry_tightening = 10.0
            insight.reasoning.append(
                f"{insight.consecutive_losses} consecutive losses: "
                "system should pause trading for at least 10 days."
            )
            return insight

        # Rule: 3+ consecutive losses -> tighten + reduce
        if insight.consecutive_losses >= 3:
            insight.position_size_mult = 0.75
            insight.cooldown_adjustment = 2
            insight.entry_tightening = 5.0
            insight.confidence_score = max(0.0, 0.5 - 0.1 * (insight.consecutive_losses - 3))
            insight.reasoning.append(
                f"{insight.consecutive_losses} consecutive losses: "
                "tightening IVR entry by +5, reducing size to 0.75x, "
                "+2 cooldown days."
            )

        # Rule: post-pause cautious sizing
        elif self._pause_active:
            insight.position_size_mult = 0.5
            remaining = self.post_pause_cautious_count - self._trades_since_pause
            insight.confidence_score = 0.3
            insight.reasoning.append(
                f"Post-pause cautious mode: 0.5x sizing for "
                f"{remaining} more trade(s)."
            )

        # Rule: high win rate -> allow upsize
        elif insight.recent_win_rate > 0.70 and len(recent) >= self.lookback:
            insight.position_size_mult = 1.25
            insight.confidence_score = min(1.0, 0.6 + insight.recent_win_rate * 0.4)
            insight.reasoning.append(
                f"Recent win rate {insight.recent_win_rate:.0%} over last "
                f"{len(recent)} trades: allowing 1.25x sizing."
            )

        else:
            # Neutral
            insight.position_size_mult = 1.0
            insight.confidence_score = 0.5
            insight.reasoning.append("Performance within normal range; no adjustments.")

        # Clamp sizing to [0.5, 1.5]
        insight.position_size_mult = max(0.5, min(1.5, insight.position_size_mult))

        return insight

    # ── Internal helpers ───────────────────────────────────────────────

    def _compute_pnl_trend(self) -> str:
        """Compare average P&L of last 5 trades vs the prior 5.

        Returns:
            "improving"     if last-5 avg > prior-5 avg by > 10%
            "deteriorating" if last-5 avg < prior-5 avg by > 10%
            "stable"        otherwise
        """
        if len(self._trades) < 10:
            return "stable"

        last_5 = self._trades[-5:]
        prior_5 = self._trades[-10:-5]

        avg_last = sum(t.exit_pnl for t in last_5) / 5.0
        avg_prior = sum(t.exit_pnl for t in prior_5) / 5.0

        # Use absolute scale to avoid division-by-zero on near-zero avgs
        diff = avg_last - avg_prior
        scale = max(abs(avg_prior), abs(avg_last), 1.0)

        if diff / scale > 0.10:
            return "improving"
        elif diff / scale < -0.10:
            return "deteriorating"
        return "stable"

    def _detect_patterns(self, insight: ReflectionInsight,
                         recent: List) -> None:
        """Tag common loss patterns from the recent trade window.

        Mutates ``insight.lesson_tags`` and ``insight.reasoning``.
        """
        losses = [t for t in recent if t.exit_pnl <= 0]
        wins = [t for t in recent if t.exit_pnl > 0]

        if not losses:
            return

        # Pattern: hold_too_long
        dte_exit_losses = [t for t in losses if t.exit_reason == "DTE_EXIT"]
        if len(dte_exit_losses) > len(losses) / 2:
            insight.lesson_tags.append("hold_too_long")
            insight.reasoning.append(
                f"{len(dte_exit_losses)}/{len(losses)} losses exited via "
                "DTE_EXIT: consider earlier profit-taking or stop-loss."
            )

        # Pattern: iv_mean_reversion_risk
        #   If the trade entered at high IV but IV presumably dropped (we
        #   approximate by checking hv_at_entry < iv_at_entry as a proxy
        #   for realised vol being lower than implied).
        iv_mr_losses = [
            t for t in losses
            if t.iv_at_entry - t.hv_at_entry > self.iv_drop_threshold
        ]
        if len(iv_mr_losses) > len(losses) / 2:
            insight.lesson_tags.append("iv_mean_reversion_risk")
            insight.reasoning.append(
                f"{len(iv_mr_losses)}/{len(losses)} losses had IV "
                "significantly above HV at entry: IV mean-reversion "
                "may not have compensated for directional risk."
            )

        # Pattern: cut_losers_faster
        if wins:
            avg_hold_losses = sum(t.holding_days for t in losses) / len(losses)
            avg_hold_wins = sum(t.holding_days for t in wins) / len(wins)
            if avg_hold_losses > avg_hold_wins:
                insight.lesson_tags.append("cut_losers_faster")
                insight.reasoning.append(
                    f"Avg holding for losses ({avg_hold_losses:.1f}d) exceeds "
                    f"wins ({avg_hold_wins:.1f}d): consider tighter stop-loss."
                )

        # Pattern: directional_risk
        dir_risk_losses = []
        for t in losses:
            if t.underlying_entry > 0:
                move_pct = abs(t.underlying_exit - t.underlying_entry) / t.underlying_entry
                if move_pct >= self.directional_move_pct:
                    dir_risk_losses.append(t)
        if len(dir_risk_losses) > len(losses) / 2:
            insight.lesson_tags.append("directional_risk")
            insight.reasoning.append(
                f"{len(dir_risk_losses)}/{len(losses)} losses saw "
                f"underlying move >{self.directional_move_pct:.0%}: "
                "consider delta-hedging or narrower wings."
            )

    def _flag_underperforming_regimes(self, insight: ReflectionInsight) -> None:
        """Add lesson tags for vol regimes with notably poor performance."""
        for regime in ("high", "medium", "low"):
            perf = self.get_regime_performance(regime)
            if perf.trade_count >= 3 and perf.win_rate < 0.35:
                tag = f"weak_regime_{regime}"
                insight.lesson_tags.append(tag)
                insight.reasoning.append(
                    f"Vol regime '{regime}' win rate is "
                    f"{perf.win_rate:.0%} over {perf.trade_count} trades: "
                    "consider avoiding or adjusting strategy in this regime."
                )

    # ── Pause management ───────────────────────────────────────────────

    def activate_pause(self) -> None:
        """Mark the journal as entering a trading pause.

        Call this when the system acts on ``should_pause=True``.  After the
        pause resolves (externally), call ``resolve_pause()`` to enter the
        cautious post-pause window.
        """
        self._pause_active = False  # Not yet in post-pause

    def resolve_pause(self) -> None:
        """Signal that the pause period has ended.

        The next ``post_pause_cautious_count`` trades recorded via
        ``record_trade`` will use cautious (0.5x) sizing.
        """
        self._pause_active = True
        self._trades_since_pause = 0


# ── Regime classification helper ───────────────────────────────────────────

def _classify_regime(iv_rank_at_entry: float) -> str:
    """Bin an IVR value into high / medium / low."""
    if iv_rank_at_entry >= 60.0:
        return "high"
    elif iv_rank_at_entry >= 30.0:
        return "medium"
    return "low"


# ── Report Formatting ──────────────────────────────────────────────────────

def format_reflection_report(insight: ReflectionInsight) -> str:
    """Produce a human-readable text report from a ReflectionInsight."""
    lines = []
    sep = "=" * 70

    lines.append(sep)
    lines.append("  TRADE JOURNAL — REFLECTION REPORT")
    lines.append(sep)
    lines.append("")

    # Streak & win rate
    lines.append("-" * 70)
    lines.append("  STREAK & RECENT PERFORMANCE")
    lines.append("-" * 70)
    if insight.consecutive_losses > 0:
        lines.append(f"  Current Streak:    {insight.consecutive_losses} consecutive LOSSES")
    elif insight.consecutive_wins > 0:
        lines.append(f"  Current Streak:    {insight.consecutive_wins} consecutive WINS")
    else:
        lines.append("  Current Streak:    N/A")
    lines.append(f"  Recent Win Rate:   {insight.recent_win_rate:.1%}")
    lines.append(f"  Avg P&L Trend:     {insight.avg_pnl_trend}")
    lines.append("")

    # Adjustments
    lines.append("-" * 70)
    lines.append("  ADJUSTMENTS")
    lines.append("-" * 70)
    lines.append(f"  Confidence Score:  {insight.confidence_score:.2f}")
    lines.append(f"  Position Size:     {insight.position_size_mult:.2f}x")
    lines.append(f"  Cooldown Extra:    +{insight.cooldown_adjustment} day(s)")
    lines.append(f"  Entry Tightening:  +{insight.entry_tightening:.1f} IVR points")
    if insight.should_pause:
        lines.append("  *** TRADING PAUSED — system should stop opening new positions ***")
    lines.append("")

    # Lesson tags
    if insight.lesson_tags:
        lines.append("-" * 70)
        lines.append("  LESSON TAGS")
        lines.append("-" * 70)
        for tag in insight.lesson_tags:
            lines.append(f"    - {tag}")
        lines.append("")

    # Reasoning
    if insight.reasoning:
        lines.append("-" * 70)
        lines.append("  REASONING")
        lines.append("-" * 70)
        for idx, reason in enumerate(insight.reasoning, 1):
            lines.append(f"  {idx}. {reason}")
        lines.append("")

    lines.append(sep)
    return "\n".join(lines)
