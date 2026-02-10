"""
策略4: 双推力突破策略 - Dual Thrust Breakout

【原理 / How it works】
- 经典日内/短线突破策略
- 根据过去N天的价格波动范围计算上下轨
- 价格突破上轨 → 做多
- 价格跌破下轨 → 平仓/做空

【可调参数 / Tunable parameters】
- lookback:  回看周期 (默认4天)
- k_up:      上轨系数 (默认0.7)
- k_down:    下轨系数 (默认0.7)

【如何修改 / How to modify】
- k_up != k_down → 非对称通道，可以偏向做多或做空
- 增大lookback → 更稳定的通道，适合波动大的市场
- 增大k → 更宽的通道，更少信号
"""

from typing import List, Dict, Any
from data.market_data import MarketData
from strategies.base import Strategy, Signal


class DualThrustStrategy(Strategy):
    """
    双推力突破策略
    根据历史波动范围设定动态突破阈值
    """

    def __init__(
        self,
        lookback: int = 4,
        k_up: float = 0.7,
        k_down: float = 0.7,
    ):
        self.lookback = lookback
        self.k_up = k_up
        self.k_down = k_down

    @property
    def name(self) -> str:
        return f"Dual Thrust ({self.lookback}d, k={self.k_up}/{self.k_down})"

    @property
    def params(self) -> Dict[str, Any]:
        return {
            "lookback": self.lookback,
            "k_up": self.k_up,
            "k_down": self.k_down,
        }

    def generate_signals(self, data: MarketData) -> List[int]:
        signals = [Signal.HOLD] * len(data)
        position = 0

        for i in range(self.lookback, len(data)):
            # Calculate range from lookback period
            period_highs = data.high[i - self.lookback:i]
            period_lows = data.low[i - self.lookback:i]
            period_closes = data.close[i - self.lookback:i]

            hh = max(period_highs)  # Highest high
            ll = min(period_lows)   # Lowest low
            hc = max(period_closes) # Highest close
            lc = min(period_closes) # Lowest close

            # Dual thrust range
            range_val = max(hh - lc, hc - ll)

            # Dynamic thresholds based on today's open
            upper = data.open[i] + self.k_up * range_val
            lower = data.open[i] - self.k_down * range_val

            if position == 0:
                if data.close[i] > upper:
                    signals[i] = Signal.BUY
                    position = 1
            else:
                if data.close[i] < lower:
                    signals[i] = Signal.SELL
                    position = 0

        return signals
