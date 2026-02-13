# §7.3: Cross-Platform Relative Value

## Summary

Cross-platform arbitrage between Kalshi and Polymarket is not viable as a systematic strategy: prices are correlated at r=0.988 with a median divergence of just -1 cent, and the rare large divergences are overwhelmingly driven by contract specification mismatches rather than genuine mispricings. However, a more productive relative-value approach emerges from combining intra-platform parity violations (84% of events deviate >5c from parity) and monotonicity violations (48.4% of threshold events) with time-to-expiry decay dynamics, which show a 3.2x calibration improvement from 7d+ to the final hour. The alpha opportunity lies not in cross-platform price gaps but in timing intra-platform structural mispricings against the convergence clock.

## Component Signals

### Signal 1: Cross-Platform Price Alignment (from §4.1)

Across 178 quality-matched market pairs, Kalshi and Polymarket exhibit near-perfect agreement. The Pearson correlation is r=0.988, the median divergence is -1.0 cent (Kalshi systematically slightly lower), and 92% of pairs agree within 5 cents. Only 7 pairs (4%) diverge by more than 10 cents, and most of these are false positives driven by specification mismatches -- different time horizons, different question framing, or one market being already resolved. The category with the largest systematic divergence is Fed rate decisions (-8.4c mean), driven by a single outlier pair. Keyword-matched pairs across 151 markets show a mean divergence of just -1.2c with an MAE of 1.8c.

The implication is clear: for well-defined, liquid markets, the two platforms converge to the same price. Cross-platform arbitrage cannot survive transaction costs (Kalshi fees + Polymarket gas costs) on a typical 1-2 cent spread. The only candidates for genuine disagreement are forward-looking, unsettled markets (Arsenal EPL at -14.5c, Fed December 2025 at -29.0c), but these require manual verification and represent a tiny opportunity set.

### Signal 2: Time-to-Expiry Effects (from §3.2)

Kalshi markets become monotonically more efficient as expiry approaches. Mean absolute calibration error drops from 4.46 pp at 7d+ to 3.29 pp (3-7d), 3.19 pp (1-3d), 2.68 pp (6-24h), 1.69 pp (1-6h), and 1.41 pp in the final hour -- a 3.2x total improvement. The steepest acceleration occurs in the final 24 hours, where nearly half the total improvement is concentrated. The 7d+ and 3-7d windows show persistent miscalibration above 3 pp, particularly in the 40-80 cent mid-price range, indicating that early markets carry sustained pricing inefficiency.

This decay curve provides a timing framework for any relative-value strategy: structural mispricings are widest far from expiry (where uncertainty is highest and liquidity lowest) and compress toward resolution. Entry timing must balance the wider edge available early against the execution uncertainty and spread costs of illiquid early markets.

### Signal 3: Intra-Event Parity (from §4.2)

Of 113,443 multi-outcome events on Kalshi, 84% deviate more than 5 cents from the theoretical parity sum of 100. The deviation structure depends on outcome count: 2-outcome events are systematically underpriced (median -59c), 4-outcome events are closest to parity (median +2c), and events with 6+ outcomes exhibit escalating overround that grows roughly linearly at 25-30 cents per additional contract. A 10-outcome event carries a median overround of 106 cents.

This creates two distinct opportunity classes. In high-outcome events, selling the full "No" portfolio harvests the overround. In low-outcome events, buying the full "Yes" portfolio captures the underpricing discount. The challenge in both cases is simultaneous execution across multiple legs and the staleness of `last_price` relative to executable bid/ask levels.

### Signal 4: Monotonicity Violations (from §4.3)

Among 31,403 threshold-style events, 48.4% contain at least one monotonicity violation (a higher-threshold contract priced above a lower-threshold contract). The violation rate is most extreme in crypto daily-close families (KXBTCD: 94.4%, KXETHD: 60.4%) and equity-index families (KXNASDAQ100U: 97.9%, KXINXU: 95.7%). Individual violations reach 98 cents, though the most extreme cases are concentrated in resolved or near-resolved events where the underlying has settled between two adjacent thresholds.

These violations represent model-free arbitrage: buy the lower-threshold contract, sell the higher-threshold contract, and lock in the violation magnitude minus fees. The opportunity is concentrated in a small number of high-frequency families that produce thousands of violations daily, making it the most mechanically actionable signal in the entire analysis.

## Cross-Signal Analysis

**Does convergence accelerate near expiry?** Yes, and this interacts with both parity and monotonicity signals. The §3.2 time-to-expiry data shows that calibration error compresses by 3.2x as markets approach resolution, with the sharpest improvement in the final 24 hours. This implies that intra-event parity deviations (§4.2) and monotonicity violations (§4.3) should also narrow as expiry approaches -- the same informed-trading burst that improves calibration should close structural mispricings. The strategy implication is that far-from-expiry violations offer larger edge but carry higher execution risk, while near-expiry violations are smaller but more reliable and liquid.

**Are the same markets mispriced on both platforms?** Largely no. The cross-platform analysis (§4.1) shows that the two platforms converge to near-identical prices for well-defined markets. The intra-platform mispricings identified in §4.2 and §4.3 are structural features of Kalshi's multi-outcome event architecture (overround scaling, threshold monotonicity), not price-discovery failures that differ across platforms. Polymarket's different market structure (individual binary contracts rather than event families) means these specific violation types do not have direct cross-platform analogs.

**Where do the signals reinforce each other?** The strongest confluence occurs in Kalshi threshold families with 7+ outcomes that are more than 24 hours from expiry. These events simultaneously exhibit: (a) high overround from the §4.2 parity analysis (median deviation >58c for 7+ outcomes), (b) frequent monotonicity violations from §4.3 (60-98% violation rates in the major families), and (c) wide calibration error from §3.2 (>2.68 pp at 6-24h+). When all three signals align on the same event, the combined edge is largest and the structural basis for the mispricing is clearest.

**What role does cross-platform data play?** While not useful for direct arbitrage, cross-platform price agreement serves as a validation signal. When Kalshi and Polymarket agree on a market's price (as they do 92% of the time within 5 cents), the cross-platform consensus can be used as an anchor to identify which leg of an intra-Kalshi parity violation is mispriced. If a Kalshi contract diverges from both its Polymarket counterpart and its intra-event parity constraint, that contract is the more likely mispriced leg.

## Proposed Strategy

The recommended approach is an **intra-platform relative value strategy on Kalshi**, using cross-platform consensus as a validation layer, timed against the expiry convergence curve.

**Core positions:**

1. **Monotonicity spread trades (from §4.3).** In KXBTCD, KXNASDAQ100U, KXETHD, and KXINXU families, systematically buy the lower-threshold contract and sell the higher-threshold contract whenever a monotonicity violation exceeds 5 cents (to clear fees). Prioritize pre-resolution events with violations in the 5-30 cent range; discard 1c/99c resolved-event artifacts. Target the 1-24 hour window before expiry, where violations are most likely to be executable and convergence is fastest.

2. **Parity portfolio trades (from §4.2).** For 2-3 outcome events with price sums below 95 cents, buy "Yes" on all outcomes to capture the underpricing discount. For 10+ outcome events with price sums above 150 cents, sell "No" on all outcomes to harvest the overround. Apply a volume filter (minimum 100 contracts traded per leg) to ensure executable prices.

3. **Cross-platform validation (from §4.1).** When entering a parity or monotonicity trade, check whether the anchor leg's price agrees with its Polymarket counterpart (where a match exists). If Polymarket confirms the anchor price, confidence in the trade is higher. If Polymarket disagrees, the cross-platform divergence may itself indicate a specification mismatch or stale price -- reduce position size or skip the trade.

4. **Time-to-expiry scaling (from §3.2).** Scale position sizes inversely with time to expiry. In the 7d+ window, take smaller positions (wider edge but higher uncertainty and execution risk). In the 1-24h window, take larger positions (narrower edge but faster convergence and better liquidity). Avoid the final hour entirely -- the 1.41 pp residual mispricing is too thin to survive transaction costs.

## Key Recommendations

- **Focus capital on intra-Kalshi monotonicity violations, not cross-platform arbitrage.** The cross-platform spread (median 1c) cannot survive fees, while monotonicity violations in KXBTCD and KXNASDAQ100U families average thousands of events daily with violations in the 5-98c range. This is where the mechanically exploitable edge resides.

- **Time entries to the 1-24 hour pre-expiry window.** The §3.2 calibration data shows that the final 24 hours concentrate the steepest convergence (MAE drops from 2.68 pp to 1.41 pp). Entering monotonicity and parity trades in this window maximizes the probability that the structural mispricing resolves before expiry while maintaining enough residual edge to clear transaction costs.

- **Use cross-platform consensus as a risk filter, not an alpha source.** Polymarket prices should be treated as an independent fair-value estimate. When a Kalshi intra-event trade conflicts with the Polymarket price on a matched market, flag it for review rather than sizing up. The 92% agreement rate within 5 cents means that cross-platform disagreement is a genuine warning signal.

## Limitations

- **Execution simultaneity.** Both parity portfolio trades and monotonicity spread trades require simultaneous execution of multiple legs. Partial fills create directional exposure that may overwhelm the structural edge, particularly in illiquid markets far from expiry.
- **`last_price` vs. executable price.** All four source analyses use `last_price`, which may be stale. The violations, parity deviations, and cross-platform divergences computed from last-trade data may not exist at current bid/ask levels. Live order-book data is required to validate any trade before execution.
- **Fee structure erosion.** Kalshi's per-contract fees reduce the net edge on every leg. Monotonicity violations below approximately 5 cents and parity deviations below approximately 10 cents (for multi-leg trades) may not survive after fees. The fee schedule should be modeled explicitly per family.
- **Temporal mismatch across reports.** The four source analyses use different data snapshots and time periods. The cross-platform matching (§4.1) uses a single snapshot, while the time-to-expiry analysis (§3.2) spans 67.7 million trades over an extended period. Whether the structural patterns identified in each report coexist simultaneously in real-time markets is not established.
- **Polymarket structural differences.** Polymarket uses individual binary contracts rather than Kalshi's event-family architecture. Parity violations (§4.2) and monotonicity violations (§4.3) are Kalshi-specific structural features; whether analogous mispricings exist on Polymarket requires separate investigation.
- **Resolution-window artifacts.** Many of the most extreme monotonicity violations (§4.3) occur in events that are already resolved or settling, where the 1c/99c prices reflect settlement mechanics rather than exploitable mispricing. Filtering these out reduces the apparent opportunity set substantially.
