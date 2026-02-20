# §7.1: Filtered Market Making

## Summary
The optimal filtered market-making strategy on Kalshi combines four independent signals -- category selection, directional tilt toward NO, time-of-day restriction, and the baseline structural maker edge -- to substantially improve upon naive around-the-clock, all-category, direction-agnostic market making. By concentrating liquidity provision in high-edge categories (Entertainment, Sports, Crypto) during U.S. business hours (12:00--19:00 ET) with a systematic NO-side inventory tilt, a filtered maker can plausibly achieve a 2--3x improvement in per-trade excess return relative to the unfiltered baseline of +1.12 pp, while avoiding the near-zero-edge conditions found in Politics and overnight sessions.

## Component Signals

### Signal 1: Category Selection (from §1.3, §2.2)
Maker excess return varies by a factor of 19x across categories. Entertainment leads at +2.06 pp per trade (4.12 pp maker-taker gap), followed by Weather (+1.68 pp), Crypto (+1.35 pp), and Sports (+1.23 pp). At the other extreme, Politics offers only +0.11 pp -- likely negative after fees -- and Finance is compressed at +0.58 pp.

These findings are corroborated by the calibration analysis in §1.3: the worst-calibrated categories (Politics at 13.11 pp MAE, Entertainment at 5.34 pp, Science/Tech at 3.69 pp) are where prices deviate most from true probabilities. However, the miscalibration in Politics benefits directional traders rather than makers, because the taker population in Politics is unusually sophisticated. In contrast, Entertainment and Weather miscalibration flows directly into maker pockets because takers in those categories are casual and price-insensitive.

**Actionable filter:** Prioritize Sports (highest absolute PnL at $97.5M historically, 2.44 pp gap), Crypto (2.70 pp gap at meaningful volume), and Entertainment (4.12 pp gap, lower volume). De-prioritize or exclude Politics (+0.11 pp, likely negative net of fees) and treat Finance as marginal (+0.58 pp).

### Signal 2: Directional Tilt (from §2.3)
Maker NO positions earn +1.28 pp excess return versus +0.77 pp for YES -- a +0.52 pp differential. NO accounts for 77% of total estimated maker PnL ($91.3M of $118.8M). The NO advantage is strongest in mid-probability contracts: +2.53 pp in the 31--40c range and +2.74 pp in the 51--60c range, where taker YES bias is most pronounced.

This directional signal is mechanistically linked to the YES bias documented in §6.3: takers systematically overpay for YES outcomes, and makers who accommodate that flow by selling YES (equivalently, buying NO) capture excess return beyond the baseline spread.

**Actionable filter:** Tilt quotes and inventory toward the NO side, particularly in the 30--60c price range. In practice this means: (a) quoting tighter on the NO side to attract YES-biased taker flow, (b) widening YES-side quotes to discourage filling on the less-profitable direction, and (c) allowing NO inventory to accumulate rather than aggressively hedging back to neutral.

### Signal 3: Time-of-Day Filter (from §3.1)
Taker excess returns (the mirror of maker opportunity) are least negative during 12:00--19:00 ET (mean approximately -0.84 pp), meaning the maker edge is compressed during these hours. Conversely, the overnight and pre-market window (03:00--08:00 ET) produces the worst taker returns (-1.57 to -2.20 pp), implying the widest maker edge.

At first glance this suggests makers should prefer overnight hours. However, §3.1 notes that the excess return figures are computed from the taker perspective. For a market maker, the relevant question is not just the per-trade edge but the product of edge and fill rate. The 12:00--19:00 window offers: (a) sufficient volume for reliable fills (468K--4.1M trades per hour), (b) tighter spreads that reduce inventory risk, and (c) the presence of informed flow that ensures prices are anchored to fundamentals, reducing the risk of providing liquidity into a stale or dislocated book.

The pre-market window (06:00--08:00 ET) offers the widest per-trade edge but is extremely thin (251K--1.2M trades) with wide spreads and stale quotes. A maker quoting aggressively here risks adverse selection from the few informed participants active at that time.

**Actionable filter:** Concentrate quoting during 12:00--19:00 ET for the best risk-adjusted returns. Optionally extend to 09:00--11:00 ET and 20:00--23:00 ET with wider spreads. Avoid the 06:00--08:00 ET pre-market window entirely unless running a wide-spread, low-fill-rate strategy.

### Signal 4: Baseline Maker Edge (from §2.1)
The unfiltered, all-category, all-direction, all-hours maker edge is +1.12 pp trade-weighted. This is statistically significant at 76 of 99 price points (p < 0.05) and represents $143.8M in aggregate PnL across 67.8M trades. The edge is widest in the 31--40c and 51--60c price ranges (2.60 and 2.59 pp spreads), consistent with the directional signal concentrating in the same zone.

This baseline serves as the "floor" for an unfiltered strategy. Each filter layer above is expected to improve upon this floor by selecting the subset of trades where the maker edge is above average.

## Combined Strategy

The four signals combine into a layered filter applied to the universe of potential market-making opportunities:

1. **Category gate (Signal 1):** Only provide liquidity in Sports, Crypto, Entertainment, Weather, and Science/Tech. Exclude Politics entirely. Treat Finance as optional/conditional on fee tier.

2. **Directional tilt (Signal 2):** Within the selected categories, bias inventory toward NO. Quote the NO side at tighter spreads than the YES side, especially for contracts priced between 30c and 60c. Allow NO inventory to build up to a risk limit before hedging.

3. **Time-of-day window (Signal 3):** Restrict active quoting to 12:00--19:00 ET. Outside this window, either withdraw quotes or widen spreads significantly (e.g., 2--3x the daytime spread) to compensate for thinner liquidity and higher adverse selection risk.

4. **Price-range focus (Signals 2, 4):** Concentrate quoting in the 30--60c price range where both the baseline maker edge and the NO directional advantage are largest.

These filters are applied multiplicatively: a trade must pass all active gates to receive a tight quote. The interaction effects are likely sub-additive (each filter captures some of the same favorable population), so the combined edge is not the sum of individual improvements, but each filter independently selects for more favorable conditions.

## Expected Edge

| Strategy | Estimated Excess Return (pp) | Basis |
|---|---|---|
| Unfiltered maker (all categories, all hours, both directions) | +1.12 | §2.1 baseline |
| Category-filtered (excl. Politics, Finance) | +1.30 to +1.50 | Removing the +0.11 pp Politics drag and +0.58 pp Finance drag from the weighted average shifts the residual upward |
| + NO directional tilt | +1.60 to +2.00 | The NO excess return is +1.28 pp overall; within selected categories the NO advantage should be at least +0.3 pp incremental |
| + Time-of-day filter (12:00--19:00 ET) | +1.80 to +2.30 | Restricting to the best-performing hours adds an estimated +0.2 to +0.3 pp based on the 0.5--1.0 pp improvement in taker returns during this window translating partially to improved maker conditions |
| + Price-range focus (30--60c) | +2.00 to +2.80 | Concentrating in the highest-edge price zone (2.59--2.60 pp baseline spread) vs. the all-price average of 2.25 pp |

**Central estimate:** A fully filtered market-making strategy should achieve approximately +2.0 to +2.5 pp gross excess return per trade, representing a 1.8--2.2x improvement over the +1.12 pp unfiltered baseline. On a capital-deployed basis, this translates from approximately 1.6% gross return (unfiltered) to approximately 2.9--3.6% gross return before fees.

**Important caveat on additivity:** These filters were estimated independently. In reality, the category effect, directional effect, time-of-day effect, and price-range effect are correlated -- for example, the NO advantage may already be larger in Entertainment than in Politics, so applying both the category and directional filters does not yield a fully additive improvement. The estimates above assume approximately 50--70% additivity across filters, but empirical backtesting is needed to validate the combined edge.

## Key Recommendations

- **Recommendation 1: Exclude Politics from the market-making universe.** The +0.11 pp maker edge in Politics is almost certainly negative after any reasonable fee structure. The high miscalibration in Politics (13.11 pp MAE) benefits directional traders, not makers, because the taker population is sophisticated. Capital allocated to Politics earns nearly zero gross edge while tying up inventory in a high-information-asymmetry environment.

- **Recommendation 2: Implement a systematic NO-side tilt with a 30--60c focus zone.** The single largest incremental signal is the +0.52 pp NO advantage over YES, which rises to +1.3--1.5 pp in the 30--60c price range. The market maker should quote the NO side 1--2 ticks tighter than the YES side in this range, and tolerate NO inventory accumulation up to a defined risk budget. This exploits the persistent taker YES bias without requiring a directional view on any specific event.

- **Recommendation 3: Restrict active quoting to 12:00--19:00 ET and use wide spreads outside this window.** The time-of-day filter is the most conservative signal (estimated +0.2--0.3 pp improvement) but offers the best risk reduction: overnight and pre-market hours combine thin liquidity with the worst adverse selection. Withdrawing quotes during 06:00--08:00 ET in particular avoids the most dangerous conditions identified in §3.1.

- **Recommendation 4: Allocate capital proportionally to category edge x volume.** Sports should receive the largest allocation (high volume, solid 2.44 pp gap). Crypto should receive the second-largest (moderate volume, wide 2.70 pp gap). Entertainment and Weather are high-edge but low-volume niches that deserve allocation but will not drive portfolio-level returns. Science/Tech is similar in character. This allocation maximizes expected absolute PnL rather than per-trade edge.

- **Recommendation 5: Validate combined edge via backtesting before deployment.** The estimates in this report are derived from independently-measured signals. Before deploying capital, the combined filter should be backtested on historical trade data to measure the actual joint excess return and confirm that the sub-additivity assumptions are reasonable.

## Limitations

- **Independent estimation bias.** Each component signal was measured in isolation. The combined edge estimate assumes partial additivity (50--70%), but the true interaction structure is unknown. Category selection and directional tilt may be highly correlated (e.g., Entertainment's high maker edge may already be concentrated on the NO side), which would make the combined filter less incremental than projected.

- **No fee modeling.** All excess return estimates are gross of Kalshi's fee structure. At the maker fee tier, the net edge will be reduced. Whether the filtered strategy remains profitable net of fees depends on the specific fee schedule, which varies by volume tier and has changed over time.

- **Survivorship and selection bias.** All analyses are based on finalized (resolved) markets. Markets that were delisted, expired without resolution, or remain open are excluded. If these excluded markets systematically differ in maker profitability, the estimates would be biased.

- **No inventory risk modeling.** The NO-side tilt recommendation increases directional inventory exposure. In a sustained period where YES outcomes are more common than prices imply (i.e., the YES bias temporarily reverses), the NO-tilted maker would suffer drawdowns. Position sizing and inventory limits are essential risk management tools not addressed here.

- **Composition effects within time-of-day.** The hour-of-day analysis in §3.1 is aggregated across all categories. The optimal trading window may differ by category (e.g., Sports events may settle in the evening, shifting the best hours for Sports market making). Category-specific time-of-day analysis would refine Signal 3 but was not performed.

- **Stationarity.** All signals are measured over the full historical dataset. Market structure, participant composition, and fee schedules evolve. The maker edge, category distribution, and time-of-day patterns may shift as Kalshi matures and attracts different participant types. Periodic recalibration of the filter parameters is advisable.

- **Execution assumptions.** The analysis assumes the maker can achieve fills at the quoted price without moving the market. In practice, a market-making strategy that concentrates on the most profitable subset of trades will face competition from other makers targeting the same opportunities, potentially compressing the available edge.
