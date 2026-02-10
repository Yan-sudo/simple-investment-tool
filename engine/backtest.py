"""
回测引擎 - Backtesting Engine

模拟真实交易执行，跟踪持仓、现金、手续费。
Simulates real trading execution with position tracking, cash management, and fees.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from data.market_data import MarketData
from strategies.base import Strategy, Signal


@dataclass
class Trade:
    """单笔交易记录 / Record of a single trade."""
    entry_date: str
    entry_price: float
    exit_date: str = ""
    exit_price: float = 0.0
    shares: int = 0
    pnl: float = 0.0         # 盈亏金额
    pnl_pct: float = 0.0     # 盈亏百分比
    holding_days: int = 0


@dataclass
class BacktestResult:
    """回测结果 / Complete backtest results."""
    strategy_name: str
    params: Dict

    # Portfolio tracking
    dates: List[str] = field(default_factory=list)
    portfolio_values: List[float] = field(default_factory=list)
    cash_history: List[float] = field(default_factory=list)
    position_history: List[int] = field(default_factory=list)
    signals: List[int] = field(default_factory=list)

    # Trade log
    trades: List[Trade] = field(default_factory=list)

    # Summary stats (populated by evaluator)
    initial_capital: float = 0.0
    final_value: float = 0.0
    total_commission: float = 0.0


class BacktestEngine:
    """
    回测引擎

    使用方法 / Usage:
        engine = BacktestEngine(initial_capital=100000)
        result = engine.run(strategy, market_data)
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_rate: float = 0.001,   # 0.1% 手续费
        slippage: float = 0.001,          # 0.1% 滑点
        position_size: float = 0.95,      # 每次用95%资金建仓
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.position_size = position_size

    def run(self, strategy: Strategy, data: MarketData) -> BacktestResult:
        """
        运行回测 / Run backtest.

        Args:
            strategy: Trading strategy instance
            data: Market data to backtest on

        Returns:
            BacktestResult with full portfolio history and trade log
        """
        signals = strategy.generate_signals(data)

        result = BacktestResult(
            strategy_name=strategy.name,
            params=strategy.params,
            initial_capital=self.initial_capital,
        )

        cash = self.initial_capital
        shares = 0
        total_commission = 0.0
        current_trade: Optional[Trade] = None

        for i in range(len(data)):
            price = data.close[i]
            signal = signals[i]

            # Execute trades
            if signal == Signal.BUY and shares == 0:
                # Buy: calculate position size
                buy_price = price * (1 + self.slippage)  # Slippage
                affordable = int((cash * self.position_size) / buy_price)
                if affordable > 0:
                    cost = affordable * buy_price
                    commission = cost * self.commission_rate
                    cash -= (cost + commission)
                    shares = affordable
                    total_commission += commission

                    current_trade = Trade(
                        entry_date=data.dates[i],
                        entry_price=buy_price,
                        shares=affordable,
                    )

            elif signal == Signal.SELL and shares > 0:
                # Sell: liquidate entire position
                sell_price = price * (1 - self.slippage)  # Slippage
                revenue = shares * sell_price
                commission = revenue * self.commission_rate
                cash += (revenue - commission)
                total_commission += commission

                if current_trade:
                    current_trade.exit_date = data.dates[i]
                    current_trade.exit_price = sell_price
                    current_trade.pnl = (sell_price - current_trade.entry_price) * shares - commission
                    current_trade.pnl_pct = (sell_price - current_trade.entry_price) / current_trade.entry_price
                    # Count trading days between entry and exit
                    entry_idx = data.dates.index(current_trade.entry_date)
                    current_trade.holding_days = i - entry_idx
                    result.trades.append(current_trade)
                    current_trade = None

                shares = 0

            # Record portfolio state
            portfolio_value = cash + shares * price
            result.dates.append(data.dates[i])
            result.portfolio_values.append(round(portfolio_value, 2))
            result.cash_history.append(round(cash, 2))
            result.position_history.append(shares)
            result.signals.append(signal)

        # Close any open position at end
        if shares > 0 and len(data) > 0:
            final_price = data.close[-1] * (1 - self.slippage)
            revenue = shares * final_price
            commission = revenue * self.commission_rate
            cash += (revenue - commission)
            total_commission += commission

            if current_trade:
                current_trade.exit_date = data.dates[-1]
                current_trade.exit_price = final_price
                current_trade.pnl = (final_price - current_trade.entry_price) * shares
                entry_idx = data.dates.index(current_trade.entry_date)
                current_trade.holding_days = len(data) - 1 - entry_idx
                current_trade.pnl_pct = (final_price - current_trade.entry_price) / current_trade.entry_price
                result.trades.append(current_trade)

            result.portfolio_values[-1] = round(cash, 2)
            result.cash_history[-1] = round(cash, 2)
            result.position_history[-1] = 0

        result.final_value = result.portfolio_values[-1] if result.portfolio_values else self.initial_capital
        result.total_commission = round(total_commission, 2)

        return result
