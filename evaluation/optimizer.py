"""
参数优化器 - Parameter Optimizer

用网格搜索找到最优参数组合。
Grid search to find optimal parameter combinations.

【如何更改策略 / How to modify strategies】
1. 调整参数范围 → 改 param_grid
2. 更换优化目标 → 改 optimize_target (sharpe/return/drawdown)
3. 走样本外验证 → 用 walk_forward_optimization()
"""

import itertools
from typing import Dict, List, Any, Type
from data.market_data import MarketData
from strategies.base import Strategy
from engine.backtest import BacktestEngine, BacktestResult
from evaluation.metrics import calculate_all_metrics


def grid_search(
    strategy_class: Type[Strategy],
    param_grid: Dict[str, List[Any]],
    data: MarketData,
    engine: BacktestEngine,
    optimize_target: str = "sharpe_ratio",
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """
    网格搜索参数优化 / Grid search parameter optimization.

    Args:
        strategy_class: Strategy class (not instance)
        param_grid: Dict of {param_name: [values_to_try]}
        data: Market data for backtesting
        engine: Backtest engine instance
        optimize_target: Metric to optimize ("sharpe_ratio", "total_return", etc.)
        top_n: Return top N results

    Returns:
        List of dicts with params and metrics, sorted by target metric

    Example:
        results = grid_search(
            MACrossoverStrategy,
            {"fast_window": [5, 10, 15], "slow_window": [20, 30, 50]},
            data, engine
        )
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    results = []
    total = 1
    for v in param_values:
        total *= len(v)

    print(f"  网格搜索: {total} 种参数组合...")

    for combo in itertools.product(*param_values):
        params = dict(zip(param_names, combo))
        try:
            strategy = strategy_class(**params)
            bt_result = engine.run(strategy, data)
            metrics = calculate_all_metrics(bt_result)

            target_val = metrics.get(optimize_target, 0)
            if isinstance(target_val, str):
                continue

            results.append({
                "params": params,
                "target_value": target_val,
                "metrics": metrics,
                "strategy_name": strategy.name,
            })
        except Exception as e:
            continue

    # Sort by target metric (descending, except drawdown which should be ascending)
    reverse = optimize_target != "max_drawdown"
    results.sort(key=lambda x: x["target_value"], reverse=reverse)

    return results[:top_n]


def walk_forward_optimization(
    strategy_class: Type[Strategy],
    param_grid: Dict[str, List[Any]],
    data: MarketData,
    engine: BacktestEngine,
    n_splits: int = 5,
    train_ratio: float = 0.7,
    optimize_target: str = "sharpe_ratio",
) -> Dict[str, Any]:
    """
    滚动窗口优化 (Walk-Forward Optimization)
    防止过拟合的关键方法！

    将数据分成多个窗口:
    - 每个窗口前70%用来优化参数 (in-sample)
    - 后30%用来验证 (out-of-sample)
    - 如果样本外表现远差于样本内 → 过拟合

    Args:
        n_splits: 滚动窗口数量
        train_ratio: 训练集比例
    """
    total_bars = len(data)
    window_size = total_bars // n_splits

    in_sample_sharpes = []
    out_sample_sharpes = []
    best_params_per_fold = []

    print(f"\n  滚动窗口优化: {n_splits} 折, 窗口大小={window_size}")

    for fold in range(n_splits):
        start = fold * window_size
        end = min(start + window_size, total_bars)
        if end - start < 50:
            continue

        train_end = start + int((end - start) * train_ratio)

        train_data = data.get_slice(start, train_end)
        test_data = data.get_slice(train_end, end)

        if len(train_data) < 30 or len(test_data) < 10:
            continue

        # Optimize on training data
        best = grid_search(
            strategy_class, param_grid, train_data, engine,
            optimize_target=optimize_target, top_n=1,
        )

        if not best:
            continue

        best_params = best[0]["params"]
        in_sample_metric = best[0]["target_value"]

        # Validate on test data
        strategy = strategy_class(**best_params)
        test_result = engine.run(strategy, test_data)
        test_metrics = calculate_all_metrics(test_result)
        out_sample_metric = test_metrics.get(optimize_target, 0)
        if isinstance(out_sample_metric, str):
            out_sample_metric = 0

        in_sample_sharpes.append(in_sample_metric)
        out_sample_sharpes.append(out_sample_metric)
        best_params_per_fold.append(best_params)

        print(f"  Fold {fold+1}: In-sample={in_sample_metric:.3f}, Out-sample={out_sample_metric:.3f}")

    # Summary
    avg_in = sum(in_sample_sharpes) / len(in_sample_sharpes) if in_sample_sharpes else 0
    avg_out = sum(out_sample_sharpes) / len(out_sample_sharpes) if out_sample_sharpes else 0

    degradation = (avg_in - avg_out) / abs(avg_in) * 100 if avg_in != 0 else 0

    return {
        "avg_in_sample": round(avg_in, 4),
        "avg_out_sample": round(avg_out, 4),
        "degradation_pct": round(degradation, 1),
        "overfitting_warning": degradation > 50,
        "best_params_per_fold": best_params_per_fold,
        "n_folds": len(in_sample_sharpes),
    }


def print_optimization_results(results: List[Dict], target_name: str = "sharpe_ratio"):
    """Pretty-print grid search results."""
    lines = []
    lines.append(f"\n  参数优化结果 / Optimization Results (target: {target_name})")
    lines.append("  " + "-" * 60)

    for i, r in enumerate(results):
        lines.append(f"  #{i+1}  {target_name}={r['target_value']:.4f}")
        lines.append(f"      Params: {r['params']}")
        m = r['metrics']
        lines.append(
            f"      Return={m.get('total_return_pct','N/A')}  "
            f"MaxDD={m.get('max_drawdown_pct','N/A')}  "
            f"Trades={m.get('total_trades', 0)}  "
            f"WinRate={m.get('win_rate_pct','N/A')}"
        )
    lines.append("  " + "-" * 60)
    print("\n".join(lines))


def print_walk_forward_results(wf_results: Dict):
    """Pretty-print walk-forward optimization results."""
    lines = []
    lines.append("\n  滚动窗口验证结果 / Walk-Forward Results")
    lines.append("  " + "-" * 50)
    lines.append(f"  折数 Folds:              {wf_results['n_folds']}")
    lines.append(f"  样本内均值 In-sample:    {wf_results['avg_in_sample']:.4f}")
    lines.append(f"  样本外均值 Out-sample:   {wf_results['avg_out_sample']:.4f}")
    lines.append(f"  性能衰减 Degradation:    {wf_results['degradation_pct']:.1f}%")

    if wf_results["overfitting_warning"]:
        lines.append("  ⚠ 警告: 过拟合风险高! 样本外表现远差于样本内")
        lines.append("    建议: 减少参数数量，使用更长的数据，或更换策略逻辑")
    else:
        lines.append("  ✓ 过拟合风险较低，策略较为稳健")

    lines.append("  " + "-" * 50)
    print("\n".join(lines))
