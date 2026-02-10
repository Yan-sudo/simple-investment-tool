"""
策略基类 - Strategy Base Class

所有交易策略都继承这个基类。
All trading strategies inherit from this base class.

【如何添加新策略 / How to add a new strategy】
1. 创建新文件 strategies/my_strategy.py
2. 继承 Strategy 基类
3. 实现 generate_signals() 方法
4. 返回信号列表: 1=买入, -1=卖出, 0=持有
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from data.market_data import MarketData


class Signal:
    """Trading signal constants."""
    BUY = 1
    SELL = -1
    HOLD = 0


class Strategy(ABC):
    """
    交易策略基类 / Base class for all trading strategies.

    每个策略必须实现:
    - name: 策略名称
    - generate_signals(data): 根据市场数据生成交易信号
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称 / Strategy name."""
        pass

    @property
    def params(self) -> Dict[str, Any]:
        """返回当前参数 / Return current parameters for display."""
        return {}

    @abstractmethod
    def generate_signals(self, data: MarketData) -> List[int]:
        """
        生成交易信号 / Generate trading signals.

        Args:
            data: MarketData object with OHLCV data

        Returns:
            List of signals: 1 (buy), -1 (sell), 0 (hold)
            Length must equal len(data)
        """
        pass


def moving_average(values: List[float], window: int) -> List[float]:
    """
    计算简单移动平均线 / Simple Moving Average.
    Returns list same length as input; early values use available data.
    """
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_vals = values[start:i + 1]
        result.append(sum(window_vals) / len(window_vals))
    return result


def exponential_moving_average(values: List[float], window: int) -> List[float]:
    """
    计算指数移动平均线 / Exponential Moving Average.
    """
    result = []
    multiplier = 2.0 / (window + 1)
    ema = values[0]
    for i, val in enumerate(values):
        if i == 0:
            ema = val
        else:
            ema = (val - ema) * multiplier + ema
        result.append(ema)
    return result


def standard_deviation(values: List[float], window: int) -> List[float]:
    """计算滚动标准差 / Rolling standard deviation."""
    import math
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_vals = values[start:i + 1]
        if len(window_vals) < 2:
            result.append(0.0)
            continue
        mean = sum(window_vals) / len(window_vals)
        variance = sum((x - mean) ** 2 for x in window_vals) / len(window_vals)
        result.append(math.sqrt(variance))
    return result


def rsi(values: List[float], period: int = 14) -> List[float]:
    """
    计算RSI指标 / Relative Strength Index.
    """
    if len(values) < 2:
        return [50.0] * len(values)

    result = [50.0]  # First value is neutral
    gains = []
    losses = []

    for i in range(1, len(values)):
        change = values[i] - values[i - 1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))

        if i < period:
            avg_gain = sum(gains) / len(gains) if gains else 0
            avg_loss = sum(losses) / len(losses) if losses else 0
        else:
            recent_gains = gains[-period:]
            recent_losses = losses[-period:]
            avg_gain = sum(recent_gains) / period
            avg_loss = sum(recent_losses) / period

        if avg_loss == 0:
            result.append(100.0)
        else:
            rs = avg_gain / avg_loss
            result.append(100.0 - (100.0 / (1.0 + rs)))

    return result
