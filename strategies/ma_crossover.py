"""
策略1: 均线交叉策略 - Moving Average Crossover Strategy

【原理 / How it works】
- 快线(短期均线)上穿慢线(长期均线) → 买入信号
- 快线下穿慢线 → 卖出信号
- 这是最经典的趋势跟踪策略

【可调参数 / Tunable parameters】
- fast_window: 快线周期 (默认10天)
- slow_window: 慢线周期 (默认30天)
- use_ema:     是否用EMA代替SMA (默认False)

【如何修改 / How to modify】
- 缩短周期 → 更灵敏，更多交易，更多假信号
- 延长周期 → 更稳定，更少交易，可能错过机会
- 使用EMA → 对近期价格更敏感
"""

from typing import List, Dict, Any
from data.market_data import MarketData
from strategies.base import (
    Strategy, Signal, moving_average, exponential_moving_average
)


class MACrossoverStrategy(Strategy):
    """
    均线交叉策略
    Golden Cross (买入): 快线从下方穿越慢线
    Death Cross (卖出): 快线从上方穿越慢线
    """

    def __init__(
        self,
        fast_window: int = 10,
        slow_window: int = 30,
        use_ema: bool = False,
    ):
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.use_ema = use_ema

    @property
    def name(self) -> str:
        ma_type = "EMA" if self.use_ema else "SMA"
        return f"MA Crossover ({ma_type} {self.fast_window}/{self.slow_window})"

    @property
    def params(self) -> Dict[str, Any]:
        return {
            "fast_window": self.fast_window,
            "slow_window": self.slow_window,
            "use_ema": self.use_ema,
        }

    def generate_signals(self, data: MarketData) -> List[int]:
        ma_func = exponential_moving_average if self.use_ema else moving_average
        fast_ma = ma_func(data.close, self.fast_window)
        slow_ma = ma_func(data.close, self.slow_window)

        signals = [Signal.HOLD] * len(data)

        for i in range(1, len(data)):
            # Golden cross: fast crosses above slow
            if fast_ma[i] > slow_ma[i] and fast_ma[i - 1] <= slow_ma[i - 1]:
                signals[i] = Signal.BUY
            # Death cross: fast crosses below slow
            elif fast_ma[i] < slow_ma[i] and fast_ma[i - 1] >= slow_ma[i - 1]:
                signals[i] = Signal.SELL

        return signals
