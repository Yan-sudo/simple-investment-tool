#!/usr/bin/env python3
"""
量化交易模型 - Quantitative Trading System
==========================================

用法 / Usage:
    python main.py                      # 运行所有股票策略回测 + 对比
    python main.py --strategy ma        # 只运行均线策略
    python main.py --optimize ma        # 优化均线策略参数
    python main.py --walkforward ma     # 滚动窗口验证 (检测过拟合)
    python main.py --compare            # 对比所有股票策略

    python main.py --options            # 运行所有期权策略 (SPY)
    python main.py --options --symbol QQQ   # QQQ 期权策略
    python main.py --options --strategy ic  # 只运行 Iron Condor
    python main.py --options --compare      # 对比所有期权策略
    python main.py --options --chain        # 显示期权链

策略代号 / Strategy codes:
    股票 / Stock:
        ma        均线交叉 Moving Average Crossover
        momentum  动量/RSI Momentum
        mr        均值回归 Mean Reversion
        dt        双推力   Dual Thrust

    期权 / Options:
        ic        Iron Condor (铁鹰策略)
        vs        Vertical Spread (垂直价差)
        wheel     Wheel Strategy (轮动策略)
        straddle  Long Straddle/Strangle (跨式/宽跨式)
        adaptive  IV-Adaptive (IV自适应策略)
        butterfly Long Call Butterfly (蝶式策略)
        pcr_ic    PCR+VIX-Band Enhanced Iron Condor (PCR增强型铁鹰)
        ta        TA-Driven / MACD+RSI+BB (技术分析驱动)
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


# ============================================================
#  期权策略注册表 / Option Strategy Registry
# ============================================================
OPTION_STRATEGIES = {}  # Lazy-loaded to avoid import errors if options not needed


def _load_option_strategies():
    """Lazy import option strategies."""
    global OPTION_STRATEGIES
    if OPTION_STRATEGIES:
        return

    from options.strategies import (
        IronCondorStrategy, VerticalSpreadStrategy, WheelStrategy,
        StraddleStrategy, IVAdaptiveStrategy,
        ButterflySpreadStrategy, PCREnhancedStrategy, TADrivenStrategy,
    )

    OPTION_STRATEGIES.update({
        "ic": {
            "class": IronCondorStrategy,
            "name": "Iron Condor",
            "default_params": {
                "iv_rank_entry": 50.0, "dte_target": 45,
                "short_delta": 0.16, "wing_width": 5.0,
            },
        },
        "vs": {
            "class": VerticalSpreadStrategy,
            "name": "Vertical Spread",
            "default_params": {
                "ma_fast": 20, "ma_slow": 50,
                "iv_rank_entry": 40.0, "spread_width": 5.0,
            },
        },
        "wheel": {
            "class": WheelStrategy,
            "name": "Wheel Strategy",
            "default_params": {
                "iv_rank_entry": 30.0, "dte_target": 30,
                "put_delta": 0.30, "call_delta": 0.30,
            },
        },
        "straddle": {
            "class": StraddleStrategy,
            "name": "Long Strangle",
            "default_params": {
                "iv_rank_entry": 30.0, "dte_target": 45,
                "use_strangle": True, "strangle_width": 5.0,
            },
        },
        "adaptive": {
            "class": IVAdaptiveStrategy,
            "name": "IV-Adaptive",
            "default_params": {
                "high_iv_threshold": 60.0, "low_iv_threshold": 25.0,
                "dte_target": 30,
            },
        },
        "butterfly": {
            "class": ButterflySpreadStrategy,
            "name": "Long Call Butterfly",
            "default_params": {
                "low_iv_threshold": 10.0, "high_iv_threshold": 70.0,
                "dte_target": 35, "wing_width": 5.0,
                "profit_target": 0.75,
            },
        },
        "pcr_ic": {
            "class": PCREnhancedStrategy,
            "name": "PCR-Enhanced Iron Condor",
            "default_params": {
                "iv_rank_entry": 50.0, "dte_target": 45,
                "short_delta": 0.16, "wing_width": 5.0,
                "require_pcr": True, "require_vix_band": True,
            },
        },
        "ta": {
            "class": TADrivenStrategy,
            "name": "TA-Driven (MACD/RSI/BB)",
            "default_params": {
                "dte_target": 30, "profit_target": 0.50,
                "stop_loss": 0.50,
            },
        },
    })


# ============================================================
#  股票策略执行 / Stock Strategy Execution
# ============================================================

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


# ============================================================
#  期权策略执行 / Option Strategy Execution
# ============================================================

def run_option_strategy(key: str, data, engine):
    """Run and report a single option strategy."""
    _load_option_strategies()
    from options.backtest import generate_option_report

    info = OPTION_STRATEGIES[key]
    strategy = info["class"](**info["default_params"])
    result = engine.run(strategy, data)
    print(generate_option_report(result))
    return result


def run_option_compare(data, engine):
    """Run all option strategies and compare."""
    _load_option_strategies()
    from options.backtest import generate_option_report, compare_option_strategies

    results = []
    for key, info in OPTION_STRATEGIES.items():
        strategy = info["class"](**info["default_params"])
        result = engine.run(strategy, data)
        results.append(result)
        print(generate_option_report(result))
        print()

    print(compare_option_strategies(results))
    return results


def show_option_chain(symbol, price, date):
    """Display a sample option chain."""
    from options.option_data import generate_option_chain

    print(f"\n{'=' * 70}")
    print(f"  OPTION CHAIN: {symbol} @ ${price:.2f}  ({date})")
    print(f"{'=' * 70}")

    for dte in [7, 30, 45]:
        from datetime import datetime, timedelta
        exp_date = datetime.strptime(date, "%Y-%m-%d") + timedelta(days=dte)
        exp_str = exp_date.strftime("%Y-%m-%d")

        chain = generate_option_chain(
            symbol=symbol, underlying_price=price,
            current_date=date, expiration_date=exp_str,
            base_iv=0.20, num_strikes=8, seed=42,
        )

        print(f"\n  Expiration: {exp_str} ({dte} DTE)")
        print(f"  {'─' * 66}")
        print(f"  {'CALLS':^32} {'Strike':^6} {'PUTS':^32}")
        print(f"  {'Bid':>7} {'Ask':>7} {'Delta':>7} {'IV':>6}  {'':^6}  "
              f"{'Bid':>7} {'Ask':>7} {'Delta':>7} {'IV':>6}")
        print(f"  {'─' * 66}")

        for strike in chain.strikes:
            c = chain.get_call(strike)
            p = chain.get_put(strike)
            if c and p:
                atm = " ◀" if abs(strike - price) < (chain.strikes[1] - chain.strikes[0]) * 0.6 else ""
                print(f"  {c.bid:>7.2f} {c.ask:>7.2f} {c.delta:>+7.3f} {c.iv:>5.1%}"
                      f"  {strike:>6.0f}{atm}"
                      f"  {p.bid:>7.2f} {p.ask:>7.2f} {p.delta:>+7.3f} {p.iv:>5.1%}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="量化交易回测系统 / Quant Trading Backtester (Stock + Options)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode selection
    parser.add_argument(
        "--options", action="store_true",
        help="期权模式 / Option trading mode",
    )
    parser.add_argument(
        "--strategy", "-s",
        help="运行单个策略 / Run a single strategy (stock: ma/momentum/mr/dt, option: ic/vs/wheel/straddle/adaptive)",
    )
    parser.add_argument(
        "--optimize", "-o",
        choices=list(STRATEGIES.keys()),
        help="参数优化 / Optimize strategy parameters (stock only)",
    )
    parser.add_argument(
        "--walkforward", "-w",
        choices=list(STRATEGIES.keys()),
        help="滚动窗口验证 / Walk-forward validation (stock only)",
    )
    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help="对比所有策略 / Compare all strategies",
    )
    parser.add_argument(
        "--chain", action="store_true",
        help="显示期权链 / Show option chain (use with --options)",
    )

    # Parameters
    parser.add_argument(
        "--symbol", default="SPY",
        choices=["SPY", "QQQ"],
        help="标的资产 / Underlying (default: SPY)",
    )
    parser.add_argument(
        "--capital", type=float, default=100_000,
        help="初始资金 (default: 100000)",
    )
    parser.add_argument(
        "--commission", type=float, default=0.001,
        help="手续费率 (default: 0.001 = 0.1%%)",
    )
    parser.add_argument(
        "--regime-data", action="store_true",
        help="使用多状态数据 (牛熊震荡) / Use regime-switching data",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子 (default: 42)",
    )
    parser.add_argument(
        "--days", type=int, default=750,
        help="模拟天数 (default: 750)",
    )
    # v3.1 — Trading constraints
    parser.add_argument(
        "--long-only", action="store_true",
        help="只做多 / Long-only mode (block bearish trades)",
    )
    parser.add_argument(
        "--min-hold", type=int, default=4,
        help="最短持仓天数 / Min holding days (default: 4)",
    )
    parser.add_argument(
        "--max-hold", type=int, default=40,
        help="最长持仓天数 / Max holding days (default: 40)",
    )
    parser.add_argument(
        "--no-reflection", action="store_true",
        help="禁用反思 / Disable trade reflection",
    )
    parser.add_argument(
        "--no-sentiment", action="store_true",
        help="禁用情绪 / Disable sentiment filter",
    )

    args = parser.parse_args()

    # ── OPTION MODE ──────────────────────────────────────────
    if args.options:
        from options.backtest import OptionBacktestEngine
        from options.option_data import generate_spy_data, generate_qqq_data

        print("=" * 70)
        print("  期权交易回测系统 / Option Trading Backtester")
        print(f"  标的 / Underlying: {args.symbol}")
        print("=" * 70)

        # Generate underlying data
        if args.symbol == "QQQ":
            data = generate_qqq_data(days=args.days, seed=args.seed)
        else:
            data = generate_spy_data(days=args.days, seed=args.seed)
        print(f"\n数据: {data}")

        # Show option chain
        if args.chain:
            price = data.close[-1]
            date = str(data.dates[-1])
            show_option_chain(args.symbol, price, date)
            return

        engine = OptionBacktestEngine(
            initial_capital=args.capital,
            commission_per_contract=0.65,
            long_only=args.long_only,
            min_holding_days=args.min_hold,
            max_holding_days=args.max_hold,
            enable_reflection=not args.no_reflection,
            enable_sentiment=not args.no_sentiment,
        )

        _load_option_strategies()

        if args.strategy:
            if args.strategy not in OPTION_STRATEGIES:
                print(f"未知期权策略: {args.strategy}")
                print(f"可用策略: {', '.join(OPTION_STRATEGIES.keys())}")
                sys.exit(1)
            run_option_strategy(args.strategy, data, engine)
        elif args.compare:
            run_option_compare(data, engine)
        else:
            # Default: compare all option strategies
            run_option_compare(data, engine)

    # ── STOCK MODE ───────────────────────────────────────────
    else:
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

        if args.optimize:
            run_optimize(args.optimize, data, engine)
        elif args.walkforward:
            run_walkforward(args.walkforward, data, engine)
        elif args.strategy:
            if args.strategy not in STRATEGIES:
                print(f"未知股票策略: {args.strategy}")
                print(f"可用策略: {', '.join(STRATEGIES.keys())}")
                sys.exit(1)
            run_single_strategy(args.strategy, data, engine)
        elif args.compare:
            run_compare(data, engine)
        else:
            run_compare(data, engine)


if __name__ == "__main__":
    main()
