"""
策略2: 动量策略 - Momentum / RSI Strategy

【原理 / How it works】
- RSI < 超卖线(30) → 买入 (价格跌过头了，可能反弹)
- RSI > 超买线(70) → 卖出 (价格涨过头了，可能回调)
- 结合价格动量确认信号

【可调参数 / Tunable parameters】
- rsi_period:      RSI计算周期 (默认14)
- oversold:        超卖阈值 (默认30)
- overbought:      超买阈值 (默认70)
- momentum_window:  动量确认窗口 (默认5)

【如何修改 / How to modify】
- 降低oversold/提高overbought → 更严格，更少信号，更高胜率
- 缩短rsi_period → RSI更敏感
- 增加momentum_window → 需要更长的趋势确认
"""

from typing import List, Dict, Any
from data.market_data import MarketData
from strategies.base import Strategy, Signal, rsi


class MomentumStrategy(Strategy):
    """
    动量/RSI策略
    利用RSI超买超卖信号，结合价格动量确认
    """

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        momentum_window: int = 5,
    ):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.momentum_window = momentum_window

    @property
    def name(self) -> str:
        return f"Momentum/RSI ({self.rsi_period}, {self.oversold}/{self.overbought})"

    @property
    def params(self) -> Dict[str, Any]:
        return {
            "rsi_period": self.rsi_period,
            "oversold": self.oversold,
            "overbought": self.overbought,
            "momentum_window": self.momentum_window,
        }

    def generate_signals(self, data: MarketData) -> List[int]:
        rsi_values = rsi(data.close, self.rsi_period)
        signals = [Signal.HOLD] * len(data)

        for i in range(self.momentum_window, len(data)):
            # Price momentum: compare current price to N days ago
            price_momentum = (data.close[i] - data.close[i - self.momentum_window]) / data.close[i - self.momentum_window]

            # Buy: RSI oversold + price starting to recover
            if rsi_values[i] < self.oversold and price_momentum > -0.05:
                signals[i] = Signal.BUY

            # Sell: RSI overbought + momentum fading
            elif rsi_values[i] > self.overbought and price_momentum < 0.05:
                signals[i] = Signal.SELL

        return signals
