"""
策略3: 均值回归策略 - Mean Reversion (Bollinger Bands)

【原理 / How it works】
- 价格跌破布林带下轨 → 买入 (价格偏离均值太多，会回归)
- 价格突破布林带上轨 → 卖出
- 核心假设: 价格会围绕均值波动

【可调参数 / Tunable parameters】
- window:      布林带均线周期 (默认20)
- num_std:     标准差倍数 (默认2.0)
- exit_at_mean: 是否在回归均线时平仓 (默认True)

【如何修改 / How to modify】
- 增大num_std → 更宽的带，更少信号，但每次信号更可靠
- 缩小window → 带宽变化更快，更适合短线
- exit_at_mean=False → 持仓更久，利润可能更大但也更不确定
"""

from typing import List, Dict, Any
from data.market_data import MarketData
from strategies.base import (
    Strategy, Signal, moving_average, standard_deviation
)


class MeanReversionStrategy(Strategy):
    """
    均值回归策略 (布林带)
    价格触碰下轨买入，触碰上轨卖出，均线附近可选择平仓
    """

    def __init__(
        self,
        window: int = 20,
        num_std: float = 2.0,
        exit_at_mean: bool = True,
    ):
        self.window = window
        self.num_std = num_std
        self.exit_at_mean = exit_at_mean

    @property
    def name(self) -> str:
        return f"Mean Reversion (BB {self.window}, {self.num_std}σ)"

    @property
    def params(self) -> Dict[str, Any]:
        return {
            "window": self.window,
            "num_std": self.num_std,
            "exit_at_mean": self.exit_at_mean,
        }

    def generate_signals(self, data: MarketData) -> List[int]:
        ma = moving_average(data.close, self.window)
        std = standard_deviation(data.close, self.window)

        signals = [Signal.HOLD] * len(data)
        position = 0  # 0=空仓, 1=持有

        for i in range(self.window, len(data)):
            upper_band = ma[i] + self.num_std * std[i]
            lower_band = ma[i] - self.num_std * std[i]

            if position == 0:
                # 价格跌破下轨 → 买入
                if data.close[i] < lower_band:
                    signals[i] = Signal.BUY
                    position = 1
            else:  # position == 1
                # 价格突破上轨 → 卖出
                if data.close[i] > upper_band:
                    signals[i] = Signal.SELL
                    position = 0
                # 可选: 回归均线时平仓
                elif self.exit_at_mean and data.close[i] > ma[i]:
                    signals[i] = Signal.SELL
                    position = 0

        return signals
