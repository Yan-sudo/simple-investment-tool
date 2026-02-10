#!/usr/bin/env python3
"""
量化交易模型 - Quantitative Trading System
==========================================

用法 / Usage:
    python main.py                  # 运行所有策略回测 + 对比
    python main.py --strategy ma    # 只运行均线策略
    python main.py --optimize ma    # 优化均线策略参数
    python main.py --walkforward ma # 滚动窗口验证 (检测过拟合)
    python main.py --compare        # 对比所有策略

策略代号 / Strategy codes:
    ma        均线交叉 Moving Average Crossover
    momentum  动量/RSI Momentum
    mr        均值回归 Mean Reversion
    dt        双推力   Dual Thrust
"""

import sys
import argparse

from data.market_data import generate_synthetic_data, generate_regime_data
from strategies.ma_crossover import MACrossoverStrategy
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.dual_thrust import DualThrustStrategy
from engine.backtest import BacktestEngine
from evaluation.report import generate_report, compare_strategies
from evaluation.optimizer import (
    grid_search, walk_forward_optimization,
    print_optimization_results, print_walk_forward_results,
)


# ============================================================
#  策略注册表 / Strategy Registry
#  要添加新策略，在这里注册即可
# ============================================================
STRATEGIES = {
    "ma": {
        "class": MACrossoverStrategy,
        "default_params": {"fast_window": 10, "slow_window": 30, "use_ema": False},
        "optimize_grid": {
            "fast_window": [5, 8, 10, 15, 20],
            "slow_window": [20, 30, 40, 50, 60],
            "use_ema": [True, False],
        },
    },
    "momentum": {
        "class": MomentumStrategy,
        "default_params": {"rsi_period": 14, "oversold": 30, "overbought": 70},
        "optimize_grid": {
            "rsi_period": [7, 10, 14, 21],
            "oversold": [20, 25, 30, 35],
            "overbought": [65, 70, 75, 80],
        },
    },
    "mr": {
        "class": MeanReversionStrategy,
        "default_params": {"window": 20, "num_std": 2.0},
        "optimize_grid": {
            "window": [10, 15, 20, 30],
            "num_std": [1.5, 2.0, 2.5, 3.0],
            "exit_at_mean": [True, False],
        },
    },
    "dt": {
        "class": DualThrustStrategy,
        "default_params": {"lookback": 4, "k_up": 0.7, "k_down": 0.7},
        "optimize_grid": {
            "lookback": [3, 4, 5, 7],
            "k_up": [0.5, 0.6, 0.7, 0.8],
            "k_down": [0.5, 0.6, 0.7, 0.8],
        },
    },
}


def run_single_strategy(key: str, data, engine):
    """Run and report a single strategy."""
    info = STRATEGIES[key]
    strategy = info["class"](**info["default_params"])
    result = engine.run(strategy, data)
    print(generate_report(result))
    return result


def run_compare(data, engine):
    """Run all strategies and compare."""
    results = []
    for key, info in STRATEGIES.items():
        strategy = info["class"](**info["default_params"])
        result = engine.run(strategy, data)
        results.append(result)
        print(generate_report(result))
        print()

    print(compare_strategies(results))
    return results


def run_optimize(key: str, data, engine):
    """Grid search optimization for a strategy."""
    info = STRATEGIES[key]
    print(f"\n优化策略: {key}")
    results = grid_search(
        info["class"], info["optimize_grid"], data, engine,
        optimize_target="sharpe_ratio", top_n=10,
    )
    print_optimization_results(results)

    if results:
        print("\n使用最优参数运行回测:")
        best_strategy = info["class"](**results[0]["params"])
        best_result = engine.run(best_strategy, data)
        print(generate_report(best_result))


def run_walkforward(key: str, data, engine):
    """Walk-forward optimization to detect overfitting."""
    info = STRATEGIES[key]
    print(f"\n滚动窗口验证: {key}")
    wf_results = walk_forward_optimization(
        info["class"], info["optimize_grid"], data, engine,
        n_splits=5, train_ratio=0.7,
    )
    print_walk_forward_results(wf_results)


def main():
    parser = argparse.ArgumentParser(
        description="量化交易回测系统 / Quant Trading Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--strategy", "-s",
        choices=list(STRATEGIES.keys()),
        help="运行单个策略 / Run a single strategy",
    )
    parser.add_argument(
        "--optimize", "-o",
        choices=list(STRATEGIES.keys()),
        help="参数优化 / Optimize strategy parameters",
    )
    parser.add_argument(
        "--walkforward", "-w",
        choices=list(STRATEGIES.keys()),
        help="滚动窗口验证 / Walk-forward validation",
    )
    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help="对比所有策略 / Compare all strategies",
    )
    parser.add_argument(
        "--capital",
        type=float, default=100_000,
        help="初始资金 (default: 100000)",
    )
    parser.add_argument(
        "--commission",
        type=float, default=0.001,
        help="手续费率 (default: 0.001 = 0.1%%)",
    )
    parser.add_argument(
        "--regime-data",
        action="store_true",
        help="使用多状态数据 (牛熊震荡) / Use regime-switching data",
    )
    parser.add_argument(
        "--seed",
        type=int, default=42,
        help="随机种子 (default: 42)",
    )
    parser.add_argument(
        "--days",
        type=int, default=750,
        help="模拟天数 (default: 750)",
    )

    args = parser.parse_args()

    # Generate data
    print("=" * 60)
    print("  量化交易回测系统 / Quant Trading Backtester")
    print("=" * 60)

    if args.regime_data:
        print("\n生成多状态市场数据 (牛市→震荡→熊市→...)")
        data = generate_regime_data(days=args.days, seed=args.seed)
    else:
        print("\n生成模拟市场数据 (几何布朗运动)")
        data = generate_synthetic_data(days=args.days, seed=args.seed)
    print(f"数据: {data}")

    engine = BacktestEngine(
        initial_capital=args.capital,
        commission_rate=args.commission,
    )

    # Execute requested mode
    if args.optimize:
        run_optimize(args.optimize, data, engine)
    elif args.walkforward:
        run_walkforward(args.walkforward, data, engine)
    elif args.strategy:
        run_single_strategy(args.strategy, data, engine)
    elif args.compare:
        run_compare(data, engine)
    else:
        # Default: compare all strategies
        run_compare(data, engine)


if __name__ == "__main__":
    main()
