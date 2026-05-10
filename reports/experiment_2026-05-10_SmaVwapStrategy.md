# 回测实验报告

## 元数据

| 项目 | 内容 |
|------|------|
| 实验日期 | 2026-05-10 |
| 实验版本 | **v4 (波动率动态调仓)** |
| 策略名称 | SmaVwapStrategy |
| 文件路径 | `user_data/strategies/SmaVwapStrategy.py` |
| 配置文件 | `config_backtest.json` |
| 交易所 | Binance (现货) |
| 报告路径 | `reports/experiment_2026-05-10_SmaVwapStrategy.md` |

---

## 实验演进

| 版本 | 变更 |
|------|------|
| v1 | 现货，仅做多，max_open_trades=3，0.1% 费率 |
| v2 | 合约，多空双向，无限制持仓，0.04% 费率 |
| v3 | 现货，仅做多，无限制持仓，0.04% 费率 |
| **v4 (当前)** | **v3 + ATR波动率动态调仓** |

---

## v4 新增功能：波动率动态调仓

### 原理

使用 ATR(14) 计算每个交易对的波动率百分比（ATR / 价格 × 100），**波动率越大的品种仓位越小，波动率越小的品种仓位越大**，使每个品种的风险贡献度趋于一致。

### 品种波动率与仓位调整系数

品种的 ATR% 基准值来自预热区（前 60 根 K 线，无交易信号），无未来数据泄露。

| 品种 | 预热区 ATR% | 仓位系数 | 实际开仓 |
|------|------------|---------|---------|
| BNB/USDT | 1.19% | ×1.5 | ~150 USDT |
| BTC/USDT | 1.27% | ×1.4 | ~143 USDT |
| XRP/USDT | 1.27% | ×1.4 | ~143 USDT |
| ETH/USDT | 1.40% | ×1.3 | ~130 USDT |
| ADA/USDT | 1.66% | ×1.1 | ~109 USDT |
| DOT/USDT | 1.80% | ×1.0 | ~101 USDT |
| LINK/USDT | 2.21% | ×0.8 | ~82 USDT |
| AVAX/USDT | 2.28% | ×0.8 | ~79 USDT |
| SOL/USDT | 2.46% | ×0.7 | ~73 USDT |
| DOGE/USDT | 2.65% | ×0.7 | ~68 USDT |

全品种平均 ATR = ~1.8% 作为基准线。ATR 越低 → 仓位越大，ATR 越高 → 仓位越小。

### 核心代码

```python
def populate_indicators(self, dataframe, metadata):
    # ATR(14) 波动率（百分比）
    dataframe["atr_pct"] = ta.ATR(dataframe, timeperiod=14) / dataframe["close"] * 100
    # 只用预热区（前 60 根 K 线）计算静态波动率基准
    lookback = min(60, len(dataframe))
    avg_atr = float(dataframe["atr_pct"].iloc[:lookback].mean())
    self.atr_cache[pair] = avg_atr if not np.isnan(avg_atr) else 3.0
    ...

def custom_stake_amount(self, pair, ...) -> float:
    # 全品种平均 ATR 作为动态基准
    if self._baseline_atr is None:
        values = [v for v in self.atr_cache.values() if v > 0]
        self._baseline_atr = sum(values) / len(values)
    atr_pct = self.atr_cache.get(pair, self._baseline_atr)
    ratio = self._baseline_atr / max(atr_pct, 0.5)   # 波动大→仓位小
    ratio = min(max(ratio, 0.3), 3.0)                 # 限制 0.3x ~ 3x
    return proposed_stake * ratio
```

---

## 策略源代码（完整）

```python
import numpy as np
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta


class SmaVwapStrategy(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "4h"
    can_short = False

    stoploss = -0.15
    minimal_roi = {"0": 0.30}

    startup_candle_count: int = 60

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False

    atr_cache: dict = {}
    _baseline_atr: float | None = None

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]

        dataframe["sma50"] = ta.SMA(dataframe, timeperiod=50)
        dataframe["sma50_diff5"] = dataframe["sma50"].diff(5)
        dataframe["vwap6"] = (
            (dataframe["volume"] * (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3)
            .rolling(6)
            .sum()
        ) / dataframe["volume"].rolling(6).sum()

        dataframe["atr_pct"] = ta.ATR(dataframe, timeperiod=14) / dataframe["close"] * 100

        lookback = min(60, len(dataframe))
        avg_atr = float(dataframe["atr_pct"].iloc[:lookback].mean())
        self.atr_cache[pair] = avg_atr if not np.isnan(avg_atr) else 3.0

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["sma50"])
                & (dataframe["sma50_diff5"] > 0)
                & (dataframe["close"] > dataframe["vwap6"])
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            ((dataframe["close"] < dataframe["sma50"]) & (dataframe["volume"] > 0)),
            "exit_long",
        ] = 1
        return dataframe

    def custom_stake_amount(
        self, pair, current_time, current_rate, proposed_stake,
        min_stake, max_stake, leverage, entry_tag, side, **kwargs
    ) -> float:
        if self._baseline_atr is None:
            values = [v for v in self.atr_cache.values() if v > 0]
            self._baseline_atr = sum(values) / len(values) if values else 3.0

        atr_pct = self.atr_cache.get(pair, self._baseline_atr)
        ratio = self._baseline_atr / max(atr_pct, 0.5)
        ratio = min(max(ratio, 0.3), 3.0)

        adjusted = proposed_stake * ratio
        if min_stake is not None:
            adjusted = max(adjusted, min_stake)
        adjusted = min(adjusted, max_stake)
        return adjusted
```

---

## 回测数据范围

| 项目 | 值 |
|------|-----|
| 开始日期 | 2024-05-20 00:00:00 |
| 结束日期 | 2026-05-10 08:00:00 |
| 数据长度 | 720 天 (~2年) |
| K线数量 | 每个交易对 4,383 根 4h K线 (现货) |
| 手续费 | 0.04% |

---

## 总体结果

| 指标 | 值 |
|------|-----|
| 起始资金 | 10,000.00 USDT |
| **最终余额** | **10,607.81 USDT** |
| **总盈亏** | **+607.81 USDT (+6.08%)** |
| 总交易数 | 929 |
| 日均交易数 | 1.29 |
| 平均单笔仓位 | 107.92 USDT (原基准 100) |
| 总交易额 | 201,288.53 USDT |
| 市场变化参考 | -11.99% (同期大盘) |

---

## 四版本对比

| 指标 | v1 | v2 | v3 | **v4 (当前)** |
|------|-----|-----|-----|-----|
| 方向 | 仅做多 | 多空 | 仅做多 | **仅做多** |
| 最大持仓 | 3 | 无限制 | 无限制 | **无限制** |
| 费率 | 0.1% | 0.04% | 0.04% | **0.04%** |
| 调仓 | 固定 | 固定 | 固定 | **ATR动态** |
| 总盈亏 | +75.66 (0.76%) | +449.65 (4.50%) | +532.16 (5.32%) | **+607.81 (6.08%)** |
| **Sharpe** | 0.26 | 1.58 | 1.77 | **1.88** |
| **Sortino** | 0.74 | 4.10 | 5.60 | **6.01** |
| **Calmar** | 0.99 | 2.82 | 4.08 | **4.75** |
| **SQN** | 0.49 | 1.36 | 2.18 | **2.33** |
| **Profit Factor** | 1.09 | 1.12 | 1.30 | **1.34** |
| **CAGR** | 0.38% | 2.25% | 2.66% | **3.04%** |
| 最大回撤 | 2.03% | 4.24% | 3.46% | **3.40%** |
| 最大连续亏损 | 18 | 44 | 36 | **36** |
| 平均仓位 | 100 | 96.98 | 99.86 | **107.92** |

> v4 的各项风险指标全面优于 v3。波动率调仓在不增加回撤的前提下，将 Sharpe 从 1.77 提升至 1.88，CAGR 从 2.66% 提升至 3.04%。

---

## 按交易对分解（v3 vs v4 对比）

| 交易对 | v3 盈亏 | v4 盈亏 | 变化 | ATR% | 仓位系数 |
|--------|---------|---------|------|------|---------|
| XRP/USDT | +180.48 | **+258.27** | +77.79 | 1.27% | ×1.4 |
| BTC/USDT | +44.71 | **+63.84** | +19.13 | 1.27% | ×1.4 |
| BNB/USDT | +33.26 | **+50.84** | +17.58 | 1.19% | ×1.5 |
| ETH/USDT | +33.57 | **+43.53** | +9.96 | 1.40% | ×1.3 |
| ADA/USDT | +67.44 | **+74.06** | +6.62 | 1.66% | ×1.1 |
| LINK/USDT | +17.12 | **+14.10** | -3.02 | 2.21% | ×0.8 |
| DOT/USDT | -20.10 | **-20.35** | -0.25 | 1.80% | ×1.0 |
| AVAX/USDT | +2.98 | **+2.34** | -0.64 | 2.28% | ×0.8 |
| SOL/USDT | +51.67 | **+38.20** | -13.47 | 2.46% | ×0.7 |
| DOGE/USDT | +121.03 | **+82.99** | -38.04 | 2.65% | ×0.7 |

> XRP (ATR 最低之一的 1.27%) 被加仓，多赚了 +77.79 USDT。
> DOGE (ATR 最高的 2.65%) 被减仓，少赚了 -38.04 USDT。
> 整体效果：低波动品种加仓盈利覆盖了高波动品种减仓的损失，净增 +75.65 USDT。

---

## 按出场原因分解

| 出场理由 | 次数 | 总盈亏 USDT | 盈亏占比 | 胜率 |
|---------|------|------------|---------|------|
| **ROI 止盈 (30%)** | **49** | **+1,515.94** | **+15.16%** | **100%** |
| 强制平仓 | 8 | +37.55 | +0.38% | 100% |
| 止损 (-15%) | 8 | -124.95 | -1.25% | 0% |
| 出场信号 | 864 | -820.73 | -8.21% | 22.6% |
| **总计** | **929** | **+607.81** | **+6.08%** | **27.1%** |

---

## 详细风险指标

| 指标 | v4 |
|------|-----|
| 盈利持仓平均时长 | 6天 12小时 04分 |
| 亏损持仓平均时长 | 1天 10小时 59分 |
| 最大连续盈利 | 11 次 |
| **最大连续亏损** | **36 次** |
| 最佳单笔 | LINK +30.02% |
| 最差单笔 | BNB -15.07% |
| 最佳单日 | +150.93 USDT |
| 最差单日 | -51.50 USDT |
| 盈利天数 / 亏损天数 / 平盘 | 103 / 218 / 400 |

---

## 分析总结

### v4 改进成果

波动率调仓的核心思路是**让每个品种的风险贡献度一致**。从结果看：

1. **Sharpe 1.77 → 1.88**：风险调整收益继续提升
2. **CAGR 2.66% → 3.04%**：年化收益提升 0.38%
3. **最大回撤保持 3.40%**：未因加仓而增加回撤
4. **利润增加 +75.65 USDT**：低波动品种加仓的盈利 > 高波动品种减仓的损失

### 还剩的短板

出场信号仍然是唯一瓶颈——864 次信号出场亏损 -820.73 USDT（22.6% 胜率），靠 49 次 ROI (+1,515.94 USDT) 覆盖。

### 可能的下一步

**出场信号优化** 是所有版本共同的遗留问题，也是提升空间最大的一步。常见做法：
- 加确认：连续 2 根 K 线低于 SMA(50) 才出场
- 改用 trailing stop 动态出场
- 用更大周期 SMA(100) 作为出场线

---

*报告生成时间: 2026-05-10 14:24 UTC*
*实验版本: v4 (Long Only, 无限制持仓, 0.04% 费率, ATR波动率动态调仓)*
