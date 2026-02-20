# §7.2: Smart-Money-Informed Calibration

## Summary

Smart money addresses on Polymarket earn their outsized returns primarily by exploiting the same calibration mispricings -- especially the longshot bias -- documented on Kalshi, but they do so with far greater precision and concentration than the typical market participant. The top 1% of Polymarket addresses average +35.0 pp excess return by systematically buying cheap tokens (avg 48.6 cents) that resolve favorably at an 83.6% rate, while Kalshi longshot contracts (below 15 cents) deliver -1.19 pp to undiscriminating buyers on a contract-weighted basis. This divergence suggests that the longshot mispricing is not a blanket feature of low-priced contracts but a selective opportunity: informed participants can distinguish which longshots are genuinely underpriced from which are overpriced lottery tickets, and a calibration strategy enhanced with smart-money flow data should outperform a naive fade-all-longshots approach.

## Component Signals

### Signal 1: Smart Money Profitability (from §5.1)

The analysis of 474,745 Polymarket addresses (each with 50+ resolved trades) reveals a persistent smart money tail. The top 1% (4,748 wallets) earn a mean excess return of +35.0 pp with an average PnL of $60,442 per address, while the bottom 90% lose -1.4 pp on average. The market is approximately zero-sum overall (median excess return: -0.05 pp), confirming that smart money profits come directly at the expense of uninformed participants.

Critically, the top 20 addresses share a distinctive signature: they buy tokens at very low prices (2-9 cents) and win nearly 100% of the time. This is the footprint of informed trading on long-shot outcomes -- these addresses are purchasing deep out-of-the-money contracts on outcomes about which they possess high-confidence information. Their average trade count (218 for the top 1%) is large enough that this is not survivorship luck.

**Key metric:** Top 1% buy at 48.6 cents average price, win at 83.6% -- a 35.0 pp gap that reflects genuine informational superiority.

### Signal 2: Longshot Bias (from §1.1)

Kalshi markets exhibit a classic longshot bias: every price point from 1 to 14 cents has negative excess returns for buyers, with a contract-weighted average of -1.19 pp across 6.17 billion contracts (trade-weighted: -0.74 pp across 19 million trades). The bias peaks in the 5-14c region and is mirrored symmetrically by favorite underpricing above 85 cents (+1.20 pp contract-weighted).

A naive fade-longshot strategy (selling YES below 15 cents) earns 1.19 cents per contract with a 95.2% win rate -- a statistically robust but still execution-sensitive edge that amounts to roughly 1.27% gross return on capital per trade. The bias is concentrated in the 5-14 cent range; the very cheapest contracts (0-5 cents) show smaller mispricing (-0.60 pp contract-weighted).

**Key metric:** Aggregate longshot buyers lose -1.19 pp, but smart money on Polymarket earns +35.0 pp buying in similar price ranges, implying the aggregate bias conceals a mixture of severely overpriced and genuinely underpriced longshots.

### Signal 3: Whale Activity (from §5.2)

Whale trades (top 1% by USDC size, above $1,231) achieve a +0.73 pp excess return -- statistically significant but economically modest. The edge peaks in the Large tier (p90-p99, $89-$1,231) at +0.82 pp rather than at the whale extreme, suggesting diminishing returns from price impact or hedging noise at the very largest sizes.

The price-controlled analysis is particularly revealing for the longshot zone: within the 0.01-0.20 price range, whale trades actually perform *worst* among all size buckets at -2.75 pp excess return, versus -0.49 pp for small trades. Large bets on longshots are not well-informed on average. By contrast, whales show their strongest edge in the 0.40-0.60 and 0.60-0.80 ranges (+1.23 pp and +2.34 pp respectively).

**Key metric:** Whale trades on longshots (0.01-0.20) lose -2.75 pp, while smart money addresses (from §5.1) earn +35.0 pp buying at similar prices. This means whale size alone is a poor proxy for smart money -- the two populations overlap but are not identical.

### Signal 4: Address Archetypes (from §5.3)

Clustering 1,955,246 Polymarket addresses into behavioral archetypes reveals that Market Makers are the only cluster with positive buyer-side excess returns (+0.97 pp). Retail traders suffer the worst performance (-6.88 pp), driven by systematic selection of losing longshot outcomes -- the longshot bias manifesting at the address level. Bot/HFT addresses dominate 71.7% of volume but operate at break-even (-0.07 pp), profiting from spread rather than direction. Whales show a slight negative excess return (-0.21 pp).

Retail's -6.88 pp excess return and 22.4% win rate against a 29.3% average purchase price directly confirms that retail participants are the primary victims of the longshot bias. They buy cheap tokens that fail to resolve, subsidizing the informed participants who selectively buy the cheap tokens that do resolve.

**Key metric:** Market Makers earn +0.97 pp; Retail loses -6.88 pp; the gap (~7.85 pp) represents the transfer from uninformed to informed flow, concentrated in the same longshot price region where calibration mispricings are largest.

## Cross-Signal Analysis

### Do Smart Money Addresses Exploit Longshot Markets?

The evidence strongly suggests yes, but with crucial selectivity. The top Polymarket addresses (§5.1) earn their highest returns by buying tokens at 2-9 cents -- precisely the price range where the Kalshi longshot bias (§1.1) is most pronounced. However, the *direction* of their edge is opposite to what the aggregate bias would predict. The aggregate longshot buyer loses money (Kalshi: -1.19 pp across the longshot zone; Polymarket Retail: -6.88 pp), but smart money earns enormous returns in the same zone. This implies:

1. **The longshot zone is a mixture distribution.** Some longshot contracts are genuinely overpriced (the majority, creating the aggregate bias), while a minority are severely underpriced. Smart money identifies and concentrates in the underpriced tail.
2. **The aggregate bias is a necessary condition for smart money profits.** If all longshots were correctly priced, there would be no edge on either side. The persistent overpricing of most longshots creates a pool of counterparties (retail buyers of overpriced longshots) who subsidize the informed traders who cherry-pick the underpriced ones.
3. **Smart money does not simply \"fade longshots\" -- they selectively buy them.** The naive Kalshi strategy of shorting all longshots earns 1.19 cents per contract. Smart money earns orders of magnitude more by identifying *which* longshots to buy.

### Do Whales Differ from Smart Money?

Yes, substantially. The whale population (§5.2) and the smart money population (§5.1) are defined by different criteria -- trade size versus excess return -- and they behave differently:

- **Whales avoid longshots; smart money embraces them.** Whales average a 77.8% purchase price (buying favorites), while the top 1% smart money addresses average 48.6% and the top 20 addresses buy at 2-9 cents. Whales and smart money occupy opposite ends of the price spectrum.
- **Whale edge is modest; smart money edge is enormous.** Whales earn +0.73 pp; smart money earns +35.0 pp. The gap is nearly 50x, suggesting fundamentally different information sources.
- **Whales lose on longshots; smart money wins on longshots.** In the 0.01-0.20 range, whale trades lose -2.75 pp. Smart money's top addresses achieve 90-98 pp excess return in the same range.

The implication is that trade size is a poor signal for informed longshot activity. The most informative smart money trades are likely moderate in size but concentrated on specific longshot contracts, not identifiable by dollar magnitude alone.

### Market Makers: The Consistent Edge

Market Makers (§5.3) are the only archetype with positive buyer-side excess returns (+0.97 pp) across a broad cross-section of markets. While their edge is far smaller than the top-1% smart money addresses, they represent a more accessible and replicable signal: high maker ratio (0.82), moderate trade sizes ($85 average), and consistent activity (463 trades average). Their positive excess return likely reflects informed order placement across many markets rather than concentrated bets on specific longshots.

This contrasts with the smart money tail, which is concentrated in fewer markets and trades. The two signals are complementary: Market Maker flow indicates broad market-level mispricing direction, while smart money flow on specific longshots indicates contract-level mispricing.

## Proposed Strategy

A combined Smart-Money-Informed Calibration strategy would layer Polymarket behavioral signals onto Kalshi calibration mispricings to refine the naive fade-longshot approach from §1.1.

### Step 1: Establish the Base Calibration Edge (from §1.1)

Begin with the Kalshi longshot bias as a prior: contracts priced at 5-14 cents are overpriced by about 1.67 pp in aggregate bucketed estimates, with a full-zone contract-weighted edge of 1.19 pp. This provides a baseline edge of approximately 1.27% gross return on capital per trade for a blanket fade-longshot approach.

### Step 2: Cross-Reference with Polymarket Smart Money Flow (from §5.1)

For markets with Polymarket counterparts, monitor the activity of addresses in the top 5% of excess return (threshold: +9.5 pp). If smart money addresses are *buying* a longshot YES contract on Polymarket, this is a strong contra-indicator against fading that longshot on Kalshi. The smart money's demonstrated ability to identify genuinely underpriced longshots (83.6% win rate at 48.6 cents average price) suggests their presence signals that the specific contract is an exception to the aggregate bias.

- **Fade longshots where smart money is absent.** If no top-5% addresses are active on the Polymarket YES side of a longshot, apply the full calibration fade.
- **Avoid fading (or reverse direction) where smart money is present.** If top-5% addresses are accumulating the YES side of a Polymarket longshot, either skip the Kalshi fade or consider a modest long position.

### Step 3: Use Market Maker Flow as Confirmation (from §5.3)

Market Maker directional positions provide a secondary confirmation signal. If Market Makers (maker_ratio > 0.7, 100+ trades) are net buyers on the same side as the proposed trade, this adds conviction. Market Maker edge (+0.97 pp) is smaller than smart money edge but is more consistent and applies across a broader market cross-section.

### Step 4: Discount Whale and Retail Flow (from §5.2, §5.3)

- **Ignore whale direction on longshots.** Whale trades in the 0.01-0.20 range carry negative information (-2.75 pp), so whale activity in longshot markets should not inform positioning.
- **Fade retail flow.** Retail traders lose -6.88 pp with a strong concentration in losing longshots. Heavy retail buying of a specific longshot is a contrarian signal that supports the fade.

### Step 5: Adjust Sizing Based on Signal Confluence

| Signal Configuration | Position | Sizing |
|---|---|---|
| No smart money + retail buying longshot | Fade longshot (sell YES / buy NO on Kalshi) | Full size |
| No smart money + no retail flow | Fade longshot | Base size |
| Smart money buying + retail buying | Skip (conflicting signals) | No position |
| Smart money buying + no retail | Consider buying longshot | Small exploratory size |
| Smart money buying + Market Maker confirms | Buy longshot | Moderate size |

### Expected Edge Enhancement

The naive fade-longshot strategy earns 1.19 cents per contract on Kalshi. By filtering out the longshot contracts where Polymarket smart money has identified genuine value, the strategy avoids the tail losses from shorting correctly-priced or underpriced longshots. Conservative estimates suggest this filtering could improve the per-trade edge by 20-50% (to approximately 1.4-1.8 cents per contract) while reducing strategy variance, as the loss tail on the naive fade is disproportionately concentrated in contracts where informed traders were on the other side.

## Key Recommendations

- **Recommendation 1: Build a rolling smart money leaderboard on Polymarket and cross-reference it with Kalshi longshot markets.** Maintain a list of the top 5% addresses by rolling 90-day excess return (threshold ~+9.5 pp from §5.1). When these addresses are active on a Polymarket longshot contract, flag the corresponding Kalshi contract for exclusion from the fade-longshot strategy. This is the single highest-value integration point from the four source reports.

- **Recommendation 2: Treat whale flow and smart money flow as distinct, non-interchangeable signals.** Whale trade size is a poor proxy for informed longshot activity (whales lose -2.75 pp on longshots per §5.2). The smart money signal must be constructed from excess-return-ranked addresses, not from trade size thresholds. A system that conflates the two will degrade the calibration signal.

- **Recommendation 3: Use Market Maker flow direction as a broad calibration overlay, and Retail flow as a fade signal.** Market Maker net positioning (+0.97 pp edge) provides mild directional information across all price ranges. Retail net positioning (-6.88 pp) provides a strong contrarian signal, particularly in longshot markets where Retail's loss concentration is most acute. Both signals are implementable from the archetype classification rules defined in §5.3 (maker_ratio thresholds and trade size filters).

## Limitations

- **Cross-platform inference is the primary structural risk.** Polymarket operates on Polygon (crypto-native participants, no regulatory restrictions on contract types) while Kalshi operates as a CFTC-regulated exchange (US participants, event contracts only). The participant bases, market structures, fee regimes, and available contract types differ substantially. Smart money behavior observed on Polymarket may not transfer directly to Kalshi market dynamics.

- **Different participant populations.** Polymarket's top addresses may include crypto-native actors with information advantages specific to crypto-adjacent markets (elections, crypto regulation, meme-driven events) that are less relevant to Kalshi's broader event contract universe (weather, economic indicators, sports). The smart money signal may be domain-specific rather than universal.

- **Temporal alignment challenges.** Polymarket and Kalshi contracts on the same underlying event may have different expiry dates, resolution criteria, or price update frequencies. Smart money activity on Polymarket may lead or lag Kalshi price movements by variable amounts, making real-time cross-referencing operationally complex.

- **Smart money identification lag.** As noted in §5.1, smart money addresses can only be identified retrospectively from resolved trades. There is an inherent look-ahead bias in assuming we can identify these addresses in real time. A rolling leaderboard mitigates but does not eliminate this issue, as address performance can decay or shift over time.

- **Thin overlap in contract coverage.** Not all Kalshi markets have Polymarket equivalents, and vice versa. The strategy is only applicable to the subset of events listed on both platforms, which may be a small fraction of each platform's total market count.

- **Aggregate vs. selective edge ambiguity.** The synthesis assumes smart money selectively buys underpriced longshots while avoiding overpriced ones. This is inferred from the coexistence of smart money profits and aggregate longshot losses, but the mechanism has not been directly tested. It is possible that smart money profits come from entirely different market segments (e.g., mid-range contracts) and the longshot connection is coincidental.

- **No fee or execution modeling.** All excess return figures are gross of fees. Kalshi charges fees on trades and settlements; Polymarket has trading fees and gas costs. The 1.19 pp base edge from fading longshots on Kalshi may be partially consumed by transaction costs, and the incremental improvement from smart money filtering must exceed any additional monitoring and execution costs to be net profitable.
