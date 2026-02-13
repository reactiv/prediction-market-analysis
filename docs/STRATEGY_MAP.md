# Strategy Map

A taxonomy of trading strategies analysable with the prediction-market-analysis dataset, organised by source of edge.

---

## 1. Calibration Exploitation

Strategies that profit from systematic deviations between market price and true probability.

### 1.1 Longshot Bias

Buy "no" on low-probability events (or sell "yes"). The existing `win_rate_by_price` analysis already shows overpricing at extremes. Strategy: systematically fade contracts priced <15c or >85c.

*Fields: `yes_price`, `result`, `ticker`*

### 1.2 Favourite-Longshot Asymmetry by Side

The `ev_yes_vs_no` analysis hints that yes/no sides may not be symmetrically mispriced. Strategy: exploit directional bias (e.g. "yes" contracts may be systematically overbought).

*Fields: `taker_side`, `yes_price`, `no_price`, `result`*

### 1.3 Category-Specific Miscalibration

Some categories (sports, politics, crypto) may be more or less efficient. Strategy: concentrate in the least-calibrated categories.

*Fields: `event_ticker` → category mapping, `yes_price`, `result`*

### 1.4 Calibration Drift

The `calibration_deviation_over_time` analysis shows efficiency changes over time. Strategy: dynamically weight calibration bets toward periods/regimes where markets are least efficient.

*Fields: `created_time`, `yes_price`, `result`*

---

## 2. Market Microstructure

Strategies that exploit the mechanics of order flow, maker/taker dynamics, and spread.

### 2.1 Passive Market Making (Maker Alpha)

The dataset shows makers earn systematic excess returns over takers. Strategy: always post limit orders, never cross the spread.

*Fields: `taker_side` (infer maker as counterparty), `yes_price`, `no_price`, `result`*

### 2.2 Maker Alpha by Category

`maker_taker_returns_by_category` shows the maker edge varies by market type. Strategy: only make markets in high-edge categories.

*Fields: `event_ticker`, `taker_side`, prices, `result`*

### 2.3 Maker Direction Selection

`maker_returns_by_direction` shows asymmetry in maker returns long vs short. Strategy: preferentially post on the more profitable side of the book.

*Fields: `taker_side`, `yes_price`, `no_price`, `result`*

### 2.4 Spread Dynamics

Markets with wider `yes_bid`/`yes_ask` spreads may offer more maker profit. Strategy: rank markets by spread and prioritise the widest.

*Fields: `yes_bid`, `yes_ask`, `no_bid`, `no_ask` (Kalshi market snapshots)*

### 2.5 Trade Size Segmentation

`win_rate_by_trade_size` shows calibration varies by trade size. Strategy: size trades into the bucket with best historical edge.

*Fields: `count` (Kalshi), `maker_amount`/`taker_amount` (Polymarket)*

---

## 3. Temporal / Intraday

Strategies that exploit time-varying patterns in price, volume, or efficiency.

### 3.1 Hour-of-Day Effects

`vwap_by_hour` and `returns_by_hour` reveal intraday patterns. Strategy: trade during hours with historically favourable returns, avoid unfavourable ones.

*Fields: `created_time` → extract hour*

### 3.2 Time-to-Expiry Decay (Theta Harvesting)

As markets approach `close_time`, mispricing may compress or widen. Strategy: systematically sell overpriced contracts as expiry approaches (analogous to options theta).

*Fields: `created_time`, `close_time`, `yes_price`, `result`*

### 3.3 Early Market Inefficiency

Newly opened markets may be less efficient before price discovery completes. Strategy: trade early in a market's life when calibration is worst.

*Fields: `open_time`, `created_time`, `yes_price`, `result`*

### 3.4 Volume Regime Switching

`volume_over_time` shows volume is non-stationary. Strategy: condition other strategies on volume regime (high-volume periods may have different efficiency characteristics).

*Fields: `created_time`, `count`/`taker_amount`, `result`*

---

## 4. Cross-Market Arbitrage

Strategies that exploit price divergence across platforms or related markets.

### 4.1 Kalshi-Polymarket Price Divergence

Same events listed on both platforms at different prices. Strategy: buy cheap side, sell expensive side.

*Fields: Kalshi `ticker` <-> Polymarket `question`/`slug` (requires fuzzy matching), prices on both*

### 4.2 Intra-Event Parity

Kalshi events with multiple mutually exclusive outcomes (e.g. "Who wins the election?") should sum to 100%. Strategy: when they don't, arb the deviation.

*Fields: `event_ticker` to group related `ticker`s, `yes_price`/`last_price`*

### 4.3 Conditional/Correlated Market Arbitrage

Markets on related events (e.g. "Will X happen?" and "Will X happen before Y?") should have consistent pricing. Strategy: identify and trade inconsistencies.

*Fields: `event_ticker`, `title`, `yes_price` — requires NLP on titles*

---

## 5. Agent / Address-Based (Polymarket-specific)

Strategies that exploit the transparency of on-chain addresses.

### 5.1 Smart Money Following

Identify addresses with historically high win rates. Strategy: mirror their trades.

*Fields: `maker`, `taker` (addresses), `maker_asset_id` → outcome mapping, `result`*

### 5.2 Whale Detection

Large `maker_amount`/`taker_amount` from a single address may signal informed trading. Strategy: trade in the same direction as unusually large orders.

*Fields: `maker`, `taker`, `maker_amount`, `taker_amount`, `block_number`*

### 5.3 Address Clustering / Counterparty Analysis

Group addresses by behaviour (market makers, retail, bots). Strategy: selectively take the other side of retail flow, avoid trading against known smart addresses.

*Fields: `maker`, `taker`, `transaction_hash`, trade frequency/size patterns*

### 5.4 Contract Type Segmentation

Trades route through either `CTF Exchange` or `NegRisk CTF Exchange`. Strategy: analyse whether one venue is systematically less efficient.

*Fields: `_contract`, prices, `result`*

---

## 6. Behavioural / Sentiment

Strategies that exploit cognitive biases of market participants.

### 6.1 Contrarian Volume Spikes

Sudden volume surges may indicate crowding. Strategy: fade large unidirectional volume spikes.

*Fields: `created_time`, `count`/`taker_amount`, `taker_side`, `result`*

### 6.2 Longshot Volume Share as Sentiment Indicator

`longshot_volume_share_over_time` shows how much capital chases low-prob events. Strategy: when longshot volume share is elevated, systematically sell longshots (crowd is gambling).

*Fields: `yes_price`, `count`, `created_time`, `result`*

### 6.3 "Yes" Bias Exploitation

People psychologically prefer to buy "yes" (positive framing). Strategy: systematically sell "yes" when it appears overpriced relative to calibration.

*Fields: `taker_side`, `yes_price`, `result`*

---

## 7. Composite / Multi-Signal

Strategies that combine multiple edges.

### 7.1 Filtered Market Making

Combine maker alpha (2.1) with category selection (1.3) and time-of-day (3.1). Only make markets in inefficient categories during favourable hours.

### 7.2 Smart-Money-Informed Calibration

Combine address tracking (5.1) with calibration exploitation (1.1). Weight calibration bets more heavily when smart money is aligned.

### 7.3 Cross-Platform Relative Value

Combine cross-market arb (4.1) with temporal signals (3.2). Arb Kalshi vs Polymarket, but only when near expiry where convergence is forced.

---

## Feasibility Summary

| Strategy Class | Data Completeness | Complexity | Novel? |
|---|---|---|---|
| Calibration (1) | High — prices + outcomes exist | Low | Partially explored |
| Microstructure (2) | High — maker/taker explicit | Medium | Partially explored |
| Temporal (3) | High — timestamps throughout | Medium | Lightly explored |
| Cross-Market (4) | Medium — requires entity matching | High | Unexplored |
| Address-Based (5) | High — on-chain is complete | High | Unexplored |
| Behavioural (6) | High — volume + side data | Medium | Lightly explored |
| Composite (7) | Depends on components | High | Unexplored |

### Recommended Starting Points

The lowest-hanging fruit for new analysis:

- **3.2 Time-to-Expiry Decay** — timestamps + outcomes, simple bucketing
- **3.3 Early Market Inefficiency** — same fields, complementary angle
- **5.1 Smart Money Following** — on-chain data is complete, unexplored
- **4.2 Intra-Event Parity** — clean GROUP BY, unexplored
