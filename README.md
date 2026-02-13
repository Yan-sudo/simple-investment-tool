# simple-investment-tool

AI Vibecoding 量化交易模型 — 从零开始用 Python 标准库构建的完整量化回测系统。

现在支持 **股票策略** + **期权交易模型 (SPY/QQQ)**。

## 项目结构

```
simple-investment-tool/
├── main.py                    # 主入口 (CLI界面, 股票+期权)
├── data/
│   └── market_data.py         # 市场数据生成/加载
├── strategies/                # 股票策略
│   ├── base.py                # 策略基类 + 技术指标 (SMA/EMA/RSI)
│   ├── ma_crossover.py        # 策略1: 均线交叉
│   ├── momentum.py            # 策略2: 动量/RSI
│   ├── mean_reversion.py      # 策略3: 均值回归 (布林带)
│   └── dual_thrust.py         # 策略4: 双推力突破
├── options/                   # 期权交易模型
│   ├── pricing.py             # Black-Scholes定价 + Greeks
│   ├── option_data.py         # 期权合约/链数据模型 + 合成数据
│   ├── iv_model.py            # 隐含波动率分析 (HV/IV Rank/IV Percentile)
│   ├── strategies.py          # 5种期权策略 (Iron Condor/Wheel/垂直价差等)
│   └── backtest.py            # 期权回测引擎 + 报告生成
├── engine/
│   └── backtest.py            # 股票回测引擎
├── evaluation/
│   ├── metrics.py             # 性能指标 + 策略评级
│   ├── report.py              # 报告生成 + ASCII图表
│   └── optimizer.py           # 参数优化 + 过拟合检测
└── requirements.txt           # 依赖 (仅需Python 3.8+标准库)
```

## 快速开始

### 股票策略

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

### 期权策略 (SPY/QQQ)

```bash
# 运行所有期权策略对比 (默认SPY)
python main.py --options

# QQQ 期权策略
python main.py --options --symbol QQQ

# 运行单个期权策略
python main.py --options --strategy ic        # Iron Condor 铁鹰
python main.py --options --strategy vs        # Vertical Spread 垂直价差
python main.py --options --strategy wheel     # Wheel 轮动策略
python main.py --options --strategy straddle  # Long Strangle 宽跨式
python main.py --options --strategy adaptive  # IV-Adaptive IV自适应

# 对比所有期权策略
python main.py --options --compare

# 查看期权链 (7/30/45 DTE)
python main.py --options --chain
python main.py --options --chain --symbol QQQ
```

## 期权交易模型

### 核心组件

#### 1. Black-Scholes 定价引擎 (`options/pricing.py`)

- **European Call/Put 定价** — 标准 Black-Scholes-Merton 公式
- **全套 Greeks 计算**:
  - **Delta (Δ)**: 标的价格变动对期权价格的影响
  - **Gamma (Γ)**: Delta 的变化率
  - **Theta (Θ)**: 时间衰减 (每日)
  - **Vega (V)**: 波动率变动对价格的影响
  - **Rho (ρ)**: 利率变动对价格的影响
- **隐含波动率 (IV) 反算** — Newton-Raphson 迭代法

#### 2. IV 波动率分析 (`options/iv_model.py`)

- **Historical Volatility (HV)** — 收盘价对数收益标准差 × √252
- **Parkinson Volatility** — 基于 High-Low 的更高效估计
- **IV Rank** — 当前IV在过去一年范围中的位置 (0-100)
- **IV Percentile** — 过去一年中低于当前IV的天数百分比
- **波动率状态识别** — Low / Normal / High / Extreme
- **IV交易信号** — 基于IV Rank + IV/HV比值的自动信号

#### 3. 期权链生成 (`options/option_data.py`)

- 合成期权链，带有 **真实波动率微笑/偏斜 (Volatility Smile/Skew)**
- 支持 SPY (低波动) 和 QQQ (高波动) 特性
- 真实的 Bid-Ask Spread (ATM 紧，Wings 宽)
- 自动 Greeks 计算

### 五种期权策略

| 策略 | 代号 | 原理 | 最佳场景 |
|------|------|------|---------|
| **Iron Condor** | `ic` | 卖出OTM看跌价差 + OTM看涨价差 | 高IV + 区间震荡 |
| **Vertical Spread** | `vs` | 趋势方向的信用价差 (牛市卖看跌/熊市卖看涨) | 有方向偏好 + 中高IV |
| **Wheel Strategy** | `wheel` | 卖CSP → 被行权持股 → 卖Covered Call → 循环 | 愿意持有标的 + 稳定收入 |
| **Long Strangle** | `straddle` | 买入OTM看涨 + OTM看跌 | 低IV + 预期大波动 |
| **IV-Adaptive** | `adaptive` | 根据IV状态自动切换上述策略 | 全市场环境自适应 |

### 策略详解

**Iron Condor (铁鹰策略)**
- 在高IV环境下卖出远期虚值看跌和看涨价差
- 利用 Theta 时间衰减收取权利金
- 通过 Delta 选择短腿行权价 (~16 delta)
- 50% 最大利润止盈 / 2倍收到权利金止损

**Wheel Strategy (轮动策略)**
- Phase 1: 卖出现金担保看跌期权 (Cash-Secured Put)
- Phase 2: 如果被行权，持有股票并卖出看涨期权 (Covered Call)
- Phase 3: 如果被叫走，回到 Phase 1
- 持续降低成本基础，收取权利金收入

**IV-Adaptive (IV自适应策略)**
- IV Rank > 60 → Iron Condor (卖premium)
- IV Rank 25-60 → Vertical Spread (方向性+卖premium)
- IV Rank < 25 → Long Strangle (买premium)

### 回测指标

期权回测报告包含:

- **收益指标**: 总回报、年化回报、P&L
- **风险指标**: 最大回撤、Sharpe Ratio
- **交易统计**: 胜率、盈亏比、平均持仓天数
- **权利金分析**: 收取权利金总额 vs 支付权利金总额
- **策略评级**: 星级评分系统
- **ASCII 权益曲线**: 终端友好的图表
- **交易明细**: 每笔交易的入场/出场/P&L

## 股票策略

| 策略 | 原理 | 适合市场 |
|------|------|---------|
| MA Crossover | 快慢均线交叉 | 趋势市 |
| Momentum/RSI | RSI超买超卖 | 震荡市 |
| Mean Reversion | 布林带回归 | 震荡市 |
| Dual Thrust | 价格突破通道 | 波动大的市场 |

## 三个核心问题

### 1. 如何用 AI Vibecoding 手搓量化模型？

**Vibecoding = 用自然语言描述你要的策略，让AI写代码。** 关键步骤:

- **描述策略逻辑**: "当短期均线上穿长期均线时买入，下穿时卖出"
- **AI生成代码**: 继承 `Strategy` 基类，实现 `generate_signals()` 方法
- **迭代优化**: "加一个RSI过滤条件" / "把止损加上" / "换成EMA"

添加新股票策略只需3步:
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

添加新期权策略:
```python
# options/strategies.py
from options.strategies import OptionStrategy, OptionPosition, OptionTrade

class MyOptionStrategy(OptionStrategy):
    @property
    def name(self):
        return "My Option Strategy"

    def generate_trades(self, closes, highs, lows, dates, risk_free_rate=0.04):
        # 你的期权交易逻辑
        # 返回 List[Optional[OptionPosition]]
        pass
```

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

## 技术说明

- 纯 Python 标准库，零外部依赖
- 内置模拟数据生成器 (几何布朗运动 + 多状态切换)
- 支持从CSV加载真实数据
- 包含手续费和滑点模拟
- Black-Scholes 使用 `math.erf` 实现正态分布CDF (无需 scipy)
- Newton-Raphson 法反算隐含波动率
- 合成期权链带有真实的 Volatility Smile 和 Skew
