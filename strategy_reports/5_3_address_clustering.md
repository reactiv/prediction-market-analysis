# 5.3: Address Clustering

## Summary

Analysis of 1,955,246 Polymarket addresses classifies participants into five behavioral archetypes based on trade frequency, size, and maker/taker ratio. Market Makers (1.7% of addresses) are the only cluster with positive excess returns (+0.97 pp), while Retail traders suffer the steepest losses (-6.88 pp). Bot/HFT addresses account for just 1.0% of addresses but dominate 71.7% of total volume, operating near break-even (-0.07 pp) consistent with automated market-making strategies that profit from spread rather than directional bets.

## Methodology

**Data sources:**
- 40,454 trade parquet files spanning all Polymarket CTF Exchange activity
- 41 market parquet files covering all listed markets
- 763,168 resolved token IDs extracted from closed binary markets

**Feature extraction:** For each of the 1,955,246 unique addresses observed across all trades, the following features were computed in a single SQL pass:
- `n_trades`: total appearances as either maker or taker
- `n_as_maker` / `n_as_taker`: counts in each role
- `maker_ratio`: n_as_maker / n_trades
- `avg_amount`: average USDC amount per trade
- `total_volume`: cumulative USDC volume

**Buyer-side performance:** For resolved markets only, each address's win rate, average purchase price, and excess return (win_rate - avg_price) were computed from the buyer side of each trade. 1,926,821 addresses had at least one resolved buyer trade.

**Classification rules (applied in priority order, later rules override earlier):**

| Archetype | Rule | Rationale |
|---|---|---|
| **Retail** | maker_ratio < 0.3 AND avg_amount < $25.16 (median) | Predominantly takes orders, small size |
| **Market Maker** | maker_ratio > 0.7 AND n_trades > 100 | Primarily posts orders, active |
| **Whale** | avg_amount > $511.00 (p95) AND n_trades > 50 | Very large average trade size |
| **Bot/HFT** | n_trades > 2,504 (p99) | Extremely high frequency |
| **Other** | None of the above | Unclassified |

**Sample sizes per cluster:** Market Maker: 33,858 | Retail: 13,754 | Whale: 22,865 | Bot/HFT: 19,553 | Other: 1,865,216.

## Results

### Cluster Composition
![Cluster composition](figures/5_3_cluster_composition.png)

The vast majority of addresses (95.4%) fall into the "Other" category, reflecting casual participants who do not exhibit strong maker/taker skew, extreme size, or high frequency. Among the defined archetypes, Market Makers are the largest group (33,858 addresses, 1.7%), followed by Whales (22,865, 1.2%), Bot/HFT (19,553, 1.0%), and Retail (13,754, 0.7%).

Volume concentration is extreme. Bot/HFT addresses -- just 1.0% of addresses -- account for $50.2 billion (71.7%) of all USDC volume, averaging 31,723 trades per address. Whales contribute $7.4B (10.6%), while the 1.87 million "Other" addresses collectively produce $11.1B (15.9%). Market Makers account for $1.2B (1.7%), and Retail is negligible at $21M (0.03%).

### Performance by Cluster
![Excess returns by cluster](figures/5_3_cluster_excess_returns.png)

Market Makers are the only cluster with statistically significant positive excess returns, achieving +0.97 pp (95% CI: +/- 0.06 pp). Their average win rate (49.7%) slightly exceeds their average purchase price (48.7%), consistent with earning a small edge through spread capture and informed order placement.

Retail traders are the worst performers at -6.88 pp excess return (95% CI: +/- 0.18 pp). Their average win rate of just 22.4% against an average purchase price of 29.3% indicates they systematically buy tokens that fail to resolve in their favor -- likely drawn to low-probability long-shot outcomes that underperform their implied odds.

Whales show a small negative excess return of -0.21 pp, essentially break-even. Their balanced maker ratio (0.47) and high average trade size ($1,289) suggest a mix of informed directional betting and hedging activity that nets out near zero on average.

Bot/HFT addresses operate at -0.07 pp -- statistically indistinguishable from zero given the tight confidence interval. Their near-zero excess return is consistent with automated strategies that earn revenue from spreads and rebates rather than directional accuracy.

The "Other" cluster shows -1.38 pp excess return, representing the broad population of occasional traders who collectively subsidize the more sophisticated participants.

### Summary Table
| Cluster | Addresses | Avg Trades | Maker Ratio | Avg Amount | Win Rate | Excess Return |
|---|---|---|---|---|---|---|
| Market Maker | 33,858 (1.7%) | 462.8 | 0.822 | $85.36 | 49.7% | +0.97 pp |
| Retail | 13,754 (0.7%) | 92.7 | 0.225 | $14.38 | 22.4% | -6.88 pp |
| Whale | 22,865 (1.2%) | 232.1 | 0.470 | $1,289.31 | 57.7% | -0.21 pp |
| Bot/HFT | 19,553 (1.0%) | 31,723.1 | 0.602 | $111.97 | 51.3% | -0.07 pp |
| Other | 1,865,216 (95.4%) | 89.3 | 0.511 | $133.43 | 50.9% | -1.38 pp |

## Key Findings

- **Market Makers are the only profitable cluster on the buyer side.** With +0.97 pp excess return over 33,858 addresses averaging 463 trades each, their edge is small but statistically robust (p < 0.001). Their high maker ratio (0.82) and moderate trade sizes suggest they earn edge through order placement strategy and spread capture.

- **Retail traders are the primary losers.** At -6.88 pp excess return, Retail addresses lose nearly 7 cents per dollar of implied probability on every trade. Their low win rate (22.4%) against a 29.3% average price indicates systematic selection of losing long-shot outcomes -- the canonical "favourite-longshot bias" manifesting at the address level.

- **Bot/HFT dominates volume but breaks even directionally.** The 19,553 Bot/HFT addresses (1% of total) generate 71.7% of all volume at $50.2B. Their near-zero excess return (-0.07 pp) confirms they profit from market microstructure (spreads, rebates, speed) rather than from picking winners.

- **Whales trade near break-even despite large size.** Whale addresses average $1,289 per trade and show -0.21 pp excess return -- slightly negative but far better than Retail. Their balanced maker ratio (0.47) suggests a mix of sophisticated directional traders and hedgers, with the net effect being approximately market-neutral.

- **The "Other" cluster subsidizes everyone.** The 1.87 million unclassified addresses carry the bulk of the losses at -1.38 pp, consistent with casual participants paying the "entertainment premium" that funds Market Maker profits and Bot/HFT spread revenue.

## Strategy Implication

The clearest signal from this analysis is **fade Retail, follow Market Makers**. Market Maker addresses -- identifiable by high maker ratios and consistent activity -- are the only cluster with positive buyer-side excess returns. Their edge is modest but persistent, and monitoring their directional positions could provide a weak but reliable alpha signal.

Conversely, Retail flow (taker-dominated, small-size) is a contrarian indicator: positions favored by Retail traders underperform by nearly 7 pp, meaning the opposite side of Retail trades has a meaningful expected edge.

Bot/HFT flow is uninformative for directional purposes -- these addresses are liquidity providers, not informed speculators. Whale flow is ambiguous: their near-zero excess return means following whales provides no edge on average, though the whale cluster likely contains a mixture of informed and uninformed participants whose signals cancel out in aggregate.

A practical implementation would combine cluster membership (is this address a Market Maker?) with directional position monitoring (what are they buying?) to construct an ensemble signal that overweights Market Maker flow and underweights or inverts Retail flow.

## Limitations

- **Rule-based clustering is approximate.** The archetype boundaries (e.g., maker_ratio > 0.7 for Market Makers) are heuristic thresholds, not learned from data. Different cutoffs would produce different cluster sizes and performance statistics. A data-driven approach (k-means, GMM) could yield more nuanced segmentation.
- **Overlap between archetypes is suppressed.** The priority ordering (Bot > Whale > Market Maker > Retail) means an address can only belong to one cluster. Some high-frequency whales or whale-sized market makers may be misclassified.
- **Buyer-side only.** Excess returns measure only the profitability of token purchases. Market Makers likely earn additional returns on the sell side (providing liquidity to buyers), which this analysis does not capture.
- **No time dimension.** Address features are computed over the full history. An address that started as Retail and evolved into a Bot would be classified based on its aggregate behavior, not its current state.
- **Survivorship in token resolution.** Only markets that clearly resolved (one outcome price > 0.99, the other < 0.01) are included. Voided or ambiguously settled markets are excluded.
- **No fee modeling.** All excess returns are gross of trading fees, gas costs, and platform commissions. The Market Maker edge of +0.97 pp may be partially or fully consumed by costs.
