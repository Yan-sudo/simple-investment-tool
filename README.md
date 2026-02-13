# simple-investment-tool

AI Vibecoding 量化交易模型 — 从零开始用 Python 构建的完整量化回测 + 期权交易 + 可视化系统。

支持 **股票策略** + **期权交易模型 (SPY/QQQ)** + **Streamlit 可视化仪表盘** + **Alpaca API 实盘/模拟交易**。

## 项目结构

```
simple-investment-tool/
├── main.py                        # CLI 主入口 (股票+期权回测)
├── gui/
│   └── app.py                     # Streamlit 可视化仪表盘
├── data/
│   ├── market_data.py             # 市场数据生成/加载 (GBM + Regime)
│   └── alpaca_client.py           # Alpaca REST API 客户端 (实时行情 + 交易)
├── strategies/                    # 股票策略
│   ├── base.py                    # 策略基类 + 技术指标 (SMA/EMA/RSI)
│   ├── ma_crossover.py            # 均线交叉策略
│   ├── momentum.py                # 动量/RSI策略
│   ├── mean_reversion.py          # 均值回归 (布林带)
│   └── dual_thrust.py             # 双推力突破策略
├── options/                       # 期权交易模型
│   ├── pricing.py                 # Black-Scholes 定价 + 一阶&二阶 Greeks
│   ├── option_data.py             # 期权合约/链数据 + 合成数据 (Vol Smile)
│   ├── iv_model.py                # IV 分析 (HV/IVR/IVP/PCR/VIX Band)
│   ├── strategies.py              # 7种期权策略 (IC/VS/Wheel/Straddle/Adaptive/Butterfly/PCR-IC)
│   ├── backtest.py                # 期权回测引擎 (EV框架 + Slippage + Wash Sale)
│   ├── ev_model.py                # 期望值(EV)分析框架 (Sinclair方法)
│   ├── risk_manager.py            # Greeks 监控哨兵 (Delta/Gamma/Vega 风控)
│   └── tax_compliance.py          # 税务合规 (Wash Sale 检测)
├── engine/
│   └── backtest.py                # 股票回测引擎
├── evaluation/
│   ├── metrics.py                 # 性能指标 + 策略评级
│   ├── report.py                  # 报告生成 + ASCII图表
│   └── optimizer.py               # 参数网格搜索 + 过拟合检测
├── .streamlit/
│   └── config.toml                # 深色主题配置
└── requirements.txt               # 依赖清单
```

## 快速开始

### 可视化仪表盘 (推荐)

```bash
# 安装 GUI 依赖
pip install streamlit plotly pandas

# 启动仪表盘
streamlit run gui/app.py
```

仪表盘功能:
- **Stock Strategies**: 选择策略 → 调参 → 一键回测 → Plotly 交互式资金曲线 + 回撤图 + 交易记录
- **Option Strategies**: 7种期权策略回测 → 对比模式 → EV/成本分析 → 期权链查看器
- **Alpaca Live**: 实时账户信息 → K线行情 → 模拟下单 → 持仓监控
- **深色主题**: 专业金融终端风格 (teal + dark)

### CLI 模式 — 股票策略

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

# 多状态市场数据 (牛熊震荡切换)
python main.py --compare --regime-data

# 自定义参数
python main.py --capital 500000 --commission 0.0005 --days 1000
```

### CLI 模式 — 期权策略 (SPY/QQQ)

```bash
# 运行所有期权策略对比 (默认SPY)
python main.py --options

# QQQ 期权策略
python main.py --options --symbol QQQ

# 运行单个期权策略
python main.py --options --strategy ic         # Iron Condor 铁鹰
python main.py --options --strategy vs         # Vertical Spread 垂直价差
python main.py --options --strategy wheel      # Wheel 轮动策略
python main.py --options --strategy straddle   # Long Strangle 宽跨式
python main.py --options --strategy adaptive   # IV-Adaptive 自适应
python main.py --options --strategy butterfly  # Long Call Butterfly 蝶式
python main.py --options --strategy pcr_ic     # PCR+VIX增强铁鹰

# 对比所有期权策略
python main.py --options --compare

# 查看期权链
python main.py --options --chain
python main.py --options --chain --symbol QQQ
```

### Alpaca API 配置 (可选)

连接 Alpaca 后可用真实行情数据替代合成数据，并进行模拟盘交易。

```bash
export ALPACA_API_KEY="your-api-key"
export ALPACA_SECRET_KEY="your-secret-key"
export ALPACA_BASE_URL="https://paper-api.alpaca.markets"  # 模拟盘

# 验证连接
python -c "from data.alpaca_client import test_connection; print(test_connection())"
```

## 期权交易模型

### 核心组件

#### 1. Black-Scholes 定价引擎 (`options/pricing.py`)

- **European Call/Put 定价** — 标准 Black-Scholes-Merton 公式
- **一阶 Greeks**: Delta (Δ), Gamma (Γ), Theta (Θ), Vega (V), Rho (ρ)
- **二阶 Greeks**: Charm (δΔ/δt), Vanna (δΔ/δσ), Vomma (δVega/δσ)
- **隐含波动率反算** — Newton-Raphson 迭代法
- 纯 Python stdlib，用 `math.erf` 实现正态分布 CDF

#### 2. IV 波动率分析 (`options/iv_model.py`)

- **Historical Volatility (HV)** — 对数收益标准差 × √252
- **Parkinson Volatility** — 基于 High-Low 的高效估计
- **IV Rank / IV Percentile** — 当前IV在历史中的位置 (0-100)
- **波动率状态识别** — Low / Normal / High / Extreme
- **PCR 信号** — Put/Call Ratio 布林带突破信号 (受 PyPatel 启发)
- **VIX Band** — IV 序列的均值/上下轨分析

#### 3. EV 期望值框架 (`options/ev_model.py`)

基于 Euan Sinclair《Option Trading》方法论:
- **POP (Probability of Profit)** — 基于 Black-Scholes delta
- **EV = P(win) × avg_win - P(loss) × avg_loss**
- **Edge = market_price - theoretical_price**
- 所有交易必须通过 EV Gate 才能入场

#### 4. 风险管理 (`options/risk_manager.py`)

- **Portfolio-level Greeks 监控**: Delta / Gamma / Theta / Vega 聚合
- **Delta 突破警报**: |net_delta| > 阈值 → 对冲或平仓
- **Gamma 风险量化**: dollar_gamma = γ × S² × 0.01
- **集中度检测**: 单一持仓不超过组合最大百分比

### 七种期权策略

| 策略 | 代号 | 原理 | 最佳场景 |
|------|------|------|---------|
| **Iron Condor** | `ic` | 卖出 OTM 看跌价差 + OTM 看涨价差 | 高IV + 区间震荡 |
| **Vertical Spread** | `vs` | 趋势方向的信用价差 | 方向偏好 + 中高IV |
| **Wheel Strategy** | `wheel` | CSP → 持股 → Covered Call → 循环 | 愿意持股 + 稳定收入 |
| **Long Strangle** | `straddle` | 买入 OTM 看涨 + OTM 看跌 | 低IV + 预期大波动 |
| **IV-Adaptive** | `adaptive` | 根据 IV 状态自动切换策略 | 全市场环境自适应 |
| **Long Butterfly** | `butterfly` | Buy K-w / Sell 2×K / Buy K+w | 低-中IV + 看好区间 |
| **PCR-Enhanced IC** | `pcr_ic` | 三重过滤: IVR + PCR + VIX Band | 高确信度卖 premium |

### 策略详解

**Iron Condor (铁鹰策略)**
- 在高 IV 环境下卖出远期虚值看跌和看涨价差
- 利用 Theta 时间衰减收取权利金
- 通过 Delta 选择短腿行权价 (~16 delta)
- 50% 最大利润止盈 / 2倍收到权利金止损

**Long Butterfly (蝶式策略)**
- Buy lower call + Sell 2× middle call + Buy upper call (等距翅宽)
- 有限风险、有限收益的方向中性策略
- 利用 call-price convexity 构建 debit structure
- 适合 IVR 10-70 的中低波动环境

**PCR-Enhanced Iron Condor (PCR增强铁鹰)**
- 三重入场过滤器:
  1. IV Rank > 阈值 (premium 足够厚)
  2. PCR 布林带信号 (市场情绪确认)
  3. VIX 上轨突破 (恐慌溢价)
- 全部条件满足才开仓，大幅提高胜率

**Wheel Strategy (轮动策略)**
- Phase 1: 卖出现金担保看跌期权 (Cash-Secured Put)
- Phase 2: 被行权后持有股票并卖出看涨期权 (Covered Call)
- Phase 3: 被叫走后回到 Phase 1
- 持续降低成本基础，收取权利金收入

**IV-Adaptive (IV自适应策略)**
- IV Rank > 60 → Iron Condor (卖 premium)
- IV Rank 25-60 → Vertical Spread (方向性 + 卖 premium)
- IV Rank < 25 → Long Strangle (买 premium)

### 回测指标

期权回测报告包含:
- **收益指标**: 总回报、年化回报、Sharpe / Sortino / Calmar Ratio
- **风险指标**: 最大回撤、最大连亏次数
- **交易统计**: 胜率、盈亏比、平均持仓天数
- **EV 分析**: 平均入场 EV、平均 POP、EV 过滤拒绝数
- **成本分析**: 佣金、滑点、总交易成本拖累
- **权利金分析**: 收取 vs 支付权利金总额
- **Wash Sale 检测**: 税务合规标记

## 股票策略

| 策略 | 原理 | 适合市场 |
|------|------|---------|
| MA Crossover | 快慢均线交叉 (SMA/EMA) | 趋势市 |
| Momentum/RSI | RSI 超买超卖 | 震荡市 |
| Mean Reversion | 布林带回归 | 震荡市 |
| Dual Thrust | 价格突破通道 | 高波动市场 |

## Streamlit 可视化仪表盘

```bash
streamlit run gui/app.py
```

### 功能概览

**Stock Strategies 页面**
- 策略选择器 + 实时参数调节滑块
- Plotly 交互式资金曲线 (含 Buy & Hold 基准)
- 买卖信号标记 + 回撤子图
- 交易记录表格 (排序/筛选)
- K线图 (OHLC Candlestick)
- 策略评级面板

**Option Strategies 页面**
- 7种期权策略一键回测
- 全策略对比模式 (叠加权益曲线)
- Trade Log / Cost Analysis / EV Analysis 三标签页
- 期权链查看器 (自选 DTE，展示 delta/IV/bid-ask)

**Alpaca Live 页面**
- 账户信息 (equity / buying power / cash / status)
- 实时 K线获取 (1Min / 1Hour / 1Day) + Candlestick 图表
- 模拟盘下单 (market / limit)
- 挂单管理 + 持仓查看

## 三个核心问题

### 1. 如何用 AI Vibecoding 手搓量化模型？

**Vibecoding = 用自然语言描述你要的策略，让 AI 写代码。**

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
from options.strategies import OptionStrategy, _StrategyMixin

class MyOptionStrategy(_StrategyMixin, OptionStrategy):
    @property
    def name(self):
        return "My Option Strategy"

    def generate_trades(self, closes, highs, lows, dates, risk_free_rate=0.04):
        # 你的期权交易逻辑
        pass
```

### 2. 如何检测策略好不好？

**第一层: 核心指标**
| 指标 | 好 | 一般 | 差 |
|------|-----|------|-----|
| Sharpe Ratio 夏普比率 | >1.0 | 0.5-1.0 | <0.5 |
| Max Drawdown 最大回撤 | <10% | 10-20% | >30% |
| Win Rate 胜率 | >55% | 45-55% | <40% |
| Profit Factor 盈亏比 | >1.5 | 1.0-1.5 | <1.0 |
| Calmar Ratio 卡玛比率 | >1.0 | 0.5-1.0 | <0.5 |

**第二层: 参数稳定性**
```bash
python main.py --optimize ma   # 最优参数附近也表现好 → 策略稳健
```

**第三层: 过拟合检测**
```bash
python main.py --walkforward ma
# 样本外 vs 样本内衰减 >50% → 过拟合警告
```

### 3. 如何更改/优化策略？

**改参数:** 在 `main.py` 的策略注册表中修改 `default_params`

**改逻辑:** 在对应策略文件中修改 `generate_signals()` / `generate_trades()`

**自动优化:** `python main.py --optimize ma`

**GUI 调参:** 在 Streamlit 仪表盘侧边栏实时拖动滑块

## 技术说明

- 核心引擎: 纯 Python 标准库，零外部依赖
- GUI: Streamlit + Plotly + Pandas (可选安装)
- Alpaca 客户端: 纯 stdlib (`urllib` + `json`)，零第三方依赖
- 数据: 内置 GBM + Regime 合成数据，支持 CSV 导入，支持 Alpaca 实时数据
- 定价: Black-Scholes 用 `math.erf` 实现正态分布 CDF
- IV 反算: Newton-Raphson 迭代法
- 合成期权链: 真实 Volatility Smile + Skew
- 策略 Mixin: `_StrategyMixin` 消除重复代码 (~90 行)
- 二阶 Greeks: Charm / Vanna / Vomma 用于高级风险管理
