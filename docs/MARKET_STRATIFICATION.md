# Market Stratification

How markets in the dataset differ structurally, and which dimensions matter for strategy analysis.

---

## Dataset Scale

| Platform | Finalized Markets | Unique Events | Unique Titles |
|---|---|---|---|
| Kalshi | 7.3M | 1.05M | 5.3M |
| Polymarket | 409K | - | 362K |

Kalshi markets are stored as one row per ticker (no duplicate snapshots). Each market belongs to an `event_ticker` which groups mutually exclusive outcomes (e.g. "Who wins the NBA Finals?" has 30 tickers, one per team).

---

## Dimension 1: Market Category (Prefix)

The `event_ticker` encodes a prefix that identifies the market type. There are **3,242 unique prefixes**, but the distribution is extremely long-tailed:

| Coverage | Prefixes needed |
|---|---|
| 50% of volume | 5 |
| 80% | 33 |
| 90% | 80 |
| 95% | 159 |
| 99% | 563 |

The existing `categories.py` regex system (~560 patterns) covers ~99% of volume. The remaining ~2,700 prefixes are near-zero-volume markets.

### Calibration Error by Category

Calibration error (MAE in percentage points) varies **90x** across prefixes — the single strongest stratification signal in the dataset.

**Most efficient (MAE < 2pp):**

| Prefix | MAE | Positions | Description |
|---|---|---|---|
| NASDAQ | 0.51 | 2.7M | NASDAQ daily up/down |
| BTCD | 0.71 | 9.1M | Bitcoin daily price |
| INXU | 0.83 | 1.3M | S&P 500 up/down |
| HIGHAUS | 0.88 | 1.3M | Austin high temp |
| HIGHDEN | 0.97 | 911K | Denver high temp |
| HIGHCHI | 1.11 | 1.5M | Chicago high temp |
| ETH | 1.39 | 515K | Ethereum price |
| WTAMATCH | 1.47 | 3.3M | WTA tennis matches |
| INX | 1.56 | 565K | S&P 500 variants |
| MLBGAME | 1.68 | 8.3M | Baseball game winners |
| ATPMATCH | 1.76 | 4.3M | ATP tennis matches |

**Efficient (MAE 2-4pp):**

| Prefix | MAE | Positions | Description |
|---|---|---|---|
| HIGHNY | 2.08 | 2.1M | New York high temp |
| NCAAFGAME | 2.29 | 16.4M | College football games |
| NBAGAME | 2.35 | 10.7M | NBA game winners |
| NHLGAME | 2.44 | 3.6M | NHL game winners |
| NFLSPREAD | 2.64 | 1.3M | NFL point spreads |
| NCAAMBGAME | 2.71 | 1.9M | College basketball games |
| NFLTOTAL | 2.26 | 800K | NFL over/under totals |
| NFLGAME | 3.60 | 15.2M | NFL game winners |

**Moderate (MAE 4-10pp):**

| Prefix | MAE | Positions | Description |
|---|---|---|---|
| PGATOUR | 4.01 | 990K | PGA golf tournaments |
| EPLGAME | 4.24 | 976K | Premier League games |
| UCLGAME | 4.47 | 776K | Champions League games |
| MARMAD | 6.35 | 3.0M | March Madness |
| GDP | 7.09 | 95K | GDP figures |
| FED | 8.17 | 243K | Fed decisions |
| LLM | 8.41 | 128K | AI model benchmarks |
| UFCFIGHT | 9.21 | 990K | UFC fight winners |
| PAYROLLS | 9.06 | 104K | Payroll figures |

**Inefficient (MAE > 10pp):**

| Prefix | MAE | Positions | Description |
|---|---|---|---|
| CPIYOY | 11.61 | 219K | CPI year-over-year |
| CANADAPM | 24.62 | 33K | Canadian PM |
| POPVOTE | 29.26 | 75K | Popular vote winner |
| EMMYDSERIES | 36.00 | 29K | Emmy Best Drama |
| GRAMSOTY | 37.74 | 32K | Grammy Song of Year |
| BIDENPARDON | 44.92 | 102K | Biden pardons |
| OSCARANIMATED | 45.17 | 13K | Oscar Best Animated |

### Pattern

The efficiency gradient follows an intuitive hierarchy:

1. **Near-perfect** (MAE < 2pp): Continuous-information, high-frequency markets — finance (NASDAQ, S&P, BTC, ETH), weather, high-volume sports (MLB, tennis)
2. **Efficient** (MAE 2-4pp): High-volume binary sports with well-understood odds (NFL, NBA, NHL, college football/basketball)
3. **Moderate** (MAE 4-10pp): Lower-frequency sports (golf, soccer, UFC), multi-outcome tournaments, economic indicators
4. **Inefficient** (MAE 10-45pp): Entertainment, awards, one-off political events, niche predictions

This correlates strongly with volume: huge-volume prefixes have median MAE 2.35pp, small-volume prefixes have median MAE 11.53pp.

---

## Dimension 2: Market Duration

Market duration (open to close) varies from sub-hour to 90+ days, and is not captured by category alone.

| Duration | Markets | Volume | Typical categories |
|---|---|---|---|
| <1h | 12K | 231K | Rare |
| 1-6h | 3.4M | 700M | Crypto daily, finance hourly |
| 6-24h | 1.6M | 1.0B | Weather daily, crypto |
| 1-7d | 2.2M | 5.3B | Sports games, weekly finance |
| 7-30d | 45K | 6.3B | Sports series, political events |
| 30-90d | 13K | 1.6B | Tournament winners, long-dated politics |
| 90d+ | 6K | 2.5B | Season-long futures, presidential |

Duration matters because it defines the information arrival pattern. A "0-5% of lifecycle" trade in a 2-hour crypto market is 6 minutes in; in a 90-day political market it's 4.5 days in. These are completely different regimes.

**Gap in existing analyses**: No analysis currently stratifies by duration. The lifecycle analysis averages across all durations, which muddies the signal.

---

## Dimension 3: Outcome Count per Event

Events range from standalone binary markets to 50+ mutually exclusive outcomes.

| Outcomes per event | Events | Volume |
|---|---|---|
| 1 (standalone binary) | 731K | 290M |
| 2 (paired yes/no) | 168K | 10.6B |
| 3-5 | 133K | 1.1B |
| 6-20 | 99K | 3.4B |
| 21-50 | 24K | 1.1B |
| 50+ | 42K | 1.5B |

Multi-outcome events have different dynamics: more longshots, parity constraints (prices should sum to 100%), and less attention per individual market. The 50+ outcome bucket (42K events) includes things like "Who wins the Masters?" with 97 golfers.

---

## Dimension 4: Price Zone

Roughly equal trade counts across zones, but different economic characteristics.

| Zone | Trades | Contracts |
|---|---|---|
| Deep OTM/ITM (1-10, 90-99c) | 15.5M | 5.4B |
| Moderate (11-25, 75-89c) | 19.3M | 4.5B |
| Near-money (26-40, 60-74c) | 21.6M | 4.7B |
| At-the-money (41-59c) | 15.7M | 3.6B |

Deep OTM/ITM contracts have the highest contract count (longshot bias drives volume in cheap contracts). The calibration deviation is known to be price-dependent, but other strategies (maker alpha, lifecycle effects) should also be stratified by price zone.

---

## Dimension 5: Trade Size

Trade size is a proxy for participant sophistication.

| Size | Trades | Contracts | % of contracts |
|---|---|---|---|
| 1 contract | 4.7M | 4.7M | 0.3% |
| 2-5 | 8.2M | 29M | 1.6% |
| 6-20 | 15.0M | 189M | 1.0% |
| 21-100 | 23.6M | 1.3B | 7.4% |
| 101-1000 | 17.9M | 6.2B | 34.1% |
| 1000+ | 2.7M | 10.5B | 57.3% |

The 1000+ contract bucket is 4% of trades but 57% of contracts — almost certainly institutional/algorithmic. Small trades (1-5 contracts) are likely retail. Strategy alpha may exist only against one segment.

---

## Dimension 6: Settlement Time

`close_time` serves as a natural grouping key. 71% of events have a single settlement time across all their markets. The 29% with multiple close times are barrier-style markets that can resolve early (e.g. "Will approval rating ever exceed X%?").

The busiest settlement times see ~126 events resolving simultaneously — these are the daily-resolution markets (BTC, weather, S&P) settling at 17:00 UTC.

Settlement time also partially works for cross-platform matching: 4,909 hourly windows overlap between Kalshi and Polymarket, enabling fuzzy title matching within narrow time windows.

---

## Identifier Structure

Neither platform provides normalised category tags. All categorisation must be derived from:

**Kalshi**: The `event_ticker` encodes structure: `{KX}{PREFIX}-{DATE/PARAMS}-{OUTCOME}`. The prefix is the de facto category. The title is templated English (e.g. "{Team} at {Team} Winner?"). No other metadata fields exist.

**Polymarket**: Even less structure — opaque `id`, blockchain `condition_id`, URL `slug`, and free-text `question`. No grouping mechanism equivalent to `event_ticker`.

---

## Recommended Stratification for Strategy Analysis

Not all dimensions are equally useful for all strategies. The highest-value crosses:

| Strategy | Primary stratification | Secondary | Why |
|---|---|---|---|
| Calibration exploitation | **Category** | Price zone | 90x MAE range across categories; longshot bias is price-dependent |
| Maker alpha | **Category** | Trade size | Maker edge likely exists only against retail in inefficient categories |
| Lifecycle / time-to-expiry | **Duration** | Category | "Early" means different things for 2h vs 90d markets |
| Intra-event parity | **Outcome count** | Category | Only applies to multi-outcome events; some categories have more |
| Address-based (Polymarket) | **Outcome count** | Price zone | Smart money may concentrate in multi-outcome events where retail misprices |
| Longshot bias | **Outcome count** | Category | More longshots in 50+ outcome events |
| Time-of-day | **Category** | Duration | Sports have game-time clustering; finance is continuous |
