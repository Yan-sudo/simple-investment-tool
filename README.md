# simple-investment-tool

AI Vibecoding 量化交易模型 — 从零开始用 Python 标准库构建的完整量化回测系统。

## 项目结构

```
simple-investment-tool/
├── main.py                    # 主入口 (CLI界面)
├── data/
│   └── market_data.py         # 市场数据生成/加载
├── strategies/
│   ├── base.py                # 策略基类 + 技术指标 (SMA/EMA/RSI)
│   ├── ma_crossover.py        # 策略1: 均线交叉
│   ├── momentum.py            # 策略2: 动量/RSI
│   ├── mean_reversion.py      # 策略3: 均值回归 (布林带)
│   └── dual_thrust.py         # 策略4: 双推力突破
├── engine/
│   └── backtest.py            # 回测引擎 (模拟交易执行)
├── evaluation/
│   ├── metrics.py             # 性能指标 + 策略评级
│   ├── report.py              # 报告生成 + ASCII图表
│   └── optimizer.py           # 参数优化 + 过拟合检测
└── requirements.txt           # 依赖 (仅需Python 3.8+标准库)
```

## 快速开始

```bash
# 运行所有策略对比
python main.py

# 运行单个策略
python main.py --strategy ma         # 均线交叉
python main.py --strategy momentum   # 动量/RSI
python main.py --strategy mr         # 均值回归
python main.py --strategy dt         # 双推力突破

# 参数优化 (网格搜索)
python main.py --optimize ma

# 过拟合检测 (滚动窗口验证)
python main.py --walkforward ma

# 使用多状态市场数据 (牛熊震荡切换)
python main.py --compare --regime-data

# 自定义参数
python main.py --capital 500000 --commission 0.0005 --days 1000
```

## 三个核心问题

### 1. 如何用 AI Vibecoding 手搓量化模型？

**Vibecoding = 用自然语言描述你要的策略，让AI写代码。** 关键步骤:

- **描述策略逻辑**: "当短期均线上穿长期均线时买入，下穿时卖出"
- **AI生成代码**: 继承 `Strategy` 基类，实现 `generate_signals()` 方法
- **迭代优化**: "加一个RSI过滤条件" / "把止损加上" / "换成EMA"

添加新策略只需3步:
```python
# strategies/my_strategy.py
from strategies.base import Strategy, Signal

class MyStrategy(Strategy):
    @property
    def name(self):
        return "My Strategy"

    def generate_signals(self, data):
        signals = [Signal.HOLD] * len(data)
        # 你的交易逻辑写在这里
        return signals
```

然后在 `main.py` 的 `STRATEGIES` 字典中注册即可。

### 2. 如何检测策略好不好？

系统提供三层检测:

**第一层: 核心指标**
| 指标 | 好 | 一般 | 差 |
|------|-----|------|-----|
| 夏普比率 Sharpe | >1.0 | 0.5-1.0 | <0.5 |
| 最大回撤 MaxDD | <10% | 10-20% | >30% |
| 胜率 Win Rate | >55% | 45-55% | <40% |
| 盈亏比 Profit Factor | >1.5 | 1.0-1.5 | <1.0 |

**第二层: 参数稳定性**
```bash
python main.py --optimize ma   # 如果最优参数附近的参数也表现不错 → 策略稳健
```

**第三层: 过拟合检测**
```bash
python main.py --walkforward ma
# 如果样本外表现远差于样本内(衰减>50%) → 过拟合警告!
```

### 3. 如何更改/优化策略？

**改参数:**
```python
# main.py 中修改 default_params
"ma": {
    "default_params": {"fast_window": 5, "slow_window": 20, "use_ema": True},
}
```

**改逻辑:**
```python
# 在 strategies/ma_crossover.py 中修改 generate_signals()
# 例如: 加上成交量过滤
if fast_ma[i] > slow_ma[i] and data.volume[i] > avg_volume:
    signals[i] = Signal.BUY
```

**自动优化:**
```bash
python main.py --optimize ma   # 自动搜索最佳参数组合
```

## 四个内置策略

| 策略 | 原理 | 适合市场 |
|------|------|---------|
| MA Crossover | 快慢均线交叉 | 趋势市 |
| Momentum/RSI | RSI超买超卖 | 震荡市 |
| Mean Reversion | 布林带回归 | 震荡市 |
| Dual Thrust | 价格突破通道 | 波动大的市场 |

## 技术说明

- 纯 Python 标准库，零外部依赖
- 内置模拟数据生成器 (几何布朗运动 + 多状态切换)
- 支持从CSV加载真实数据
- 包含手续费和滑点模拟
