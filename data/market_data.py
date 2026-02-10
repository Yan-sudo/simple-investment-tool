"""
市场数据模块 - Market Data Module

生成模拟市场数据 / 从CSV加载真实数据
Generates synthetic market data or loads real data from CSV files.
"""

import csv
import math
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional


class MarketData:
    """Represents OHLCV market data for a single asset."""

    def __init__(self):
        self.dates: List[str] = []
        self.open: List[float] = []
        self.high: List[float] = []
        self.low: List[float] = []
        self.close: List[float] = []
        self.volume: List[int] = []

    def __len__(self):
        return len(self.dates)

    def __repr__(self):
        if not self.dates:
            return "MarketData(empty)"
        return (
            f"MarketData({self.dates[0]} to {self.dates[-1]}, "
            f"{len(self)} bars, "
            f"price range: {min(self.close):.2f}-{max(self.close):.2f})"
        )

    def get_slice(self, start_idx: int, end_idx: int) -> "MarketData":
        """Return a subset of the data."""
        sliced = MarketData()
        sliced.dates = self.dates[start_idx:end_idx]
        sliced.open = self.open[start_idx:end_idx]
        sliced.high = self.high[start_idx:end_idx]
        sliced.low = self.low[start_idx:end_idx]
        sliced.close = self.close[start_idx:end_idx]
        sliced.volume = self.volume[start_idx:end_idx]
        return sliced


def generate_synthetic_data(
    symbol: str = "SYNTH",
    days: int = 500,
    start_price: float = 100.0,
    volatility: float = 0.02,
    trend: float = 0.0003,
    seed: Optional[int] = None,
) -> MarketData:
    """
    生成模拟股票价格数据 (几何布朗运动 + 均值回复)
    Generate synthetic stock data using Geometric Brownian Motion with mean reversion.

    参数 / Parameters:
        symbol:      股票代号
        days:        天数
        start_price: 起始价格
        volatility:  日波动率 (0.02 = 2%)
        trend:       日均趋势 (0.0003 = ~8% 年化)
        seed:        随机种子 (可复现)
    """
    if seed is not None:
        random.seed(seed)

    data = MarketData()
    price = start_price
    base_date = datetime(2023, 1, 2)

    for i in range(days):
        current_date = base_date + timedelta(days=i)
        # Skip weekends
        if current_date.weekday() >= 5:
            continue

        # GBM with mean reversion component
        mean_reversion = -0.001 * (price - start_price) / start_price
        daily_return = trend + mean_reversion + volatility * random.gauss(0, 1)
        price *= (1 + daily_return)
        price = max(price, 1.0)  # Floor at $1

        # Generate OHLCV from close price
        intraday_vol = volatility * 0.5
        o = price * (1 + random.gauss(0, intraday_vol))
        h = max(price, o) * (1 + abs(random.gauss(0, intraday_vol)))
        l = min(price, o) * (1 - abs(random.gauss(0, intraday_vol)))
        v = int(random.gauss(1_000_000, 300_000))
        v = max(v, 100_000)

        data.dates.append(current_date.strftime("%Y-%m-%d"))
        data.open.append(round(o, 2))
        data.high.append(round(h, 2))
        data.low.append(round(l, 2))
        data.close.append(round(price, 2))
        data.volume.append(v)

    return data


def generate_regime_data(days: int = 750, seed: Optional[int] = 42) -> MarketData:
    """
    生成带有不同市场状态的数据 (牛市/熊市/震荡)
    Generate data with distinct market regimes (bull/bear/sideways).
    Useful for testing strategy robustness.
    """
    if seed is not None:
        random.seed(seed)

    data = MarketData()
    price = 100.0
    base_date = datetime(2022, 1, 3)

    # Define regimes: (trend, volatility, duration_days)
    regimes = [
        (0.001, 0.015, 150),   # 牛市 Bull
        (0.0002, 0.025, 100),  # 震荡 Sideways/volatile
        (-0.0008, 0.02, 120),  # 熊市 Bear
        (0.0012, 0.012, 130),  # 强牛 Strong bull
        (-0.0003, 0.03, 100),  # 高波动下跌 High-vol decline
        (0.0005, 0.018, 150),  # 温和上涨 Mild bull
    ]

    day_count = 0
    for trend, vol, duration in regimes:
        if day_count >= days:
            break
        for _ in range(duration):
            if day_count >= days:
                break
            current_date = base_date + timedelta(days=day_count)
            day_count += 1
            if current_date.weekday() >= 5:
                continue

            daily_return = trend + vol * random.gauss(0, 1)
            price *= (1 + daily_return)
            price = max(price, 1.0)

            intraday_vol = vol * 0.5
            o = price * (1 + random.gauss(0, intraday_vol))
            h = max(price, o) * (1 + abs(random.gauss(0, intraday_vol)))
            l = min(price, o) * (1 - abs(random.gauss(0, intraday_vol)))
            v = int(random.gauss(1_000_000, 300_000))

            data.dates.append(current_date.strftime("%Y-%m-%d"))
            data.open.append(round(o, 2))
            data.high.append(round(h, 2))
            data.low.append(round(l, 2))
            data.close.append(round(price, 2))
            data.volume.append(max(v, 100_000))

    return data


def load_csv(filepath: str) -> MarketData:
    """
    从CSV文件加载数据。期望列: Date, Open, High, Low, Close, Volume
    Load market data from a CSV file.
    Expected columns: Date, Open, High, Low, Close, Volume
    """
    data = MarketData()
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.dates.append(row["Date"])
            data.open.append(float(row["Open"]))
            data.high.append(float(row["High"]))
            data.low.append(float(row["Low"]))
            data.close.append(float(row["Close"]))
            data.volume.append(int(float(row["Volume"])))
    return data


def save_csv(data: MarketData, filepath: str):
    """Save market data to CSV."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Open", "High", "Low", "Close", "Volume"])
        for i in range(len(data)):
            writer.writerow([
                data.dates[i], data.open[i], data.high[i],
                data.low[i], data.close[i], data.volume[i]
            ])
