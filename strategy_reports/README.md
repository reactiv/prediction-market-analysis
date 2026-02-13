# Strategy Reports

Analysis of prediction market data across Kalshi (7.3M markets, 72M trades) and Polymarket (409K markets, ~388M resolved trades). Each report examines a specific trading strategy or market feature.

## Family 1: Calibration Exploitation
| Report | Strategy | Key Finding |
|--------|----------|-------------|
| [§1.1](1_1_longshot_bias.md) | Longshot Bias | Contracts below 15c deliver -0.74 pp excess return to buyers; fading longshots earns 0.74 cents/contract across 19M trades with a 93.9% win rate. |
| [§1.2](1_2_favourite_longshot_asymmetry.md) | Favourite-Longshot Asymmetry | YES takers lose -1.77 pp vs -1.25 pp for NO takers, a persistent 0.52 pp gap driven by YES overpricing in the 41-70c range. |
| [§1.3](1_3_category_specific_miscalibration.md) | Category-Specific Miscalibration | Politics is the most miscalibrated category (MAE 6.21 pp), while Crypto (0.74 pp) and Sports (1.06 pp) are well-calibrated -- a 19x variation across categories. |
| [§1.4](1_4_calibration_drift.md) | Calibration Drift | Kalshi MAD fell from 18% at launch to a 0.4% plateau by mid-2022, spiked to 3.6% during the Nov 2024 election shock, and has recovered to ~1.06% by Nov 2025. |

## Family 2: Market Microstructure
| Report | Strategy | Key Finding |
|--------|----------|-------------|
| [§2.1](2_1_maker_alpha.md) | Maker Alpha | Makers earn +1.12 pp excess return vs -1.12 pp for takers at 80/99 price points, totalling ~$144M maker PnL across $17.2B volume. |
| [§2.2](2_2_maker_alpha_by_category.md) | Maker Alpha by Category | Entertainment offers the widest maker edge (+2.06 pp, 4.12 pp gap) while Politics is near-zero (+0.11 pp) -- a 19x range across 8 categories. |
| [§2.3](2_3_maker_direction_selection.md) | Maker Direction Selection | Maker NO positions earn +1.28 pp vs +0.77 pp for YES (+0.52 pp differential), accounting for 77% of total maker PnL ($91.3M of $118.8M). |
| [§2.4](2_4_spread_dynamics.md) | Spread Dynamics | Mean spread is 58.3c (median 78c) across 334K snapshots; Politics (9.7c) and Science/Tech (9.2c) are tight while the long tail is highly illiquid. |
| [§2.5](2_5_trade_size_segmentation.md) | Trade Size Segmentation | Makers earn positive excess returns against all trade sizes (+0.86 to +1.18 pp); the largest trades (1K+) show the smallest taker deficit, contradicting the "smart money" hypothesis on Kalshi. |

## Family 3: Temporal / Intraday
| Report | Strategy | Key Finding |
|--------|----------|-------------|
| [§3.1](3_1_hour_of_day_effects.md) | Hour-of-Day Effects | Taker excess returns range from -2.20 pp (08:00 ET) to -0.63 pp (noon ET), a 1.57 pp intraday spread aligned with U.S. equity market hours. |
| [§3.2](3_2_time_to_expiry_decay.md) | Time-to-Expiry Decay | Calibration error drops 3.2x from 4.46 pp (7d+ out) to 1.41 pp in the final hour, with the steepest improvement in the last 24 hours. |
| [§3.3](3_3_early_market_inefficiency.md) | Early Market Inefficiency | Peak miscalibration (3.8 pp MAE) occurs at the 15-30% lifecycle stage -- 4x worse than the best window at 50-70% (0.9 pp) -- with a systematic YES overpricing bias in early trades. |
| [§3.4](3_4_volume_regime_switching.md) | Volume Regime Switching | High-volume markets (top tercile) achieve 1.59 pp MAE vs 4.96 pp for medium-volume; the relationship is non-monotonic, with medium-volume markets the worst calibrated. |

## Family 4: Cross-Market Arbitrage
| Report | Strategy | Key Finding |
|--------|----------|-------------|
| [§4.1](4_1_kalshi_polymarket_divergence.md) | Kalshi-Polymarket Divergence | Across 178 matched pairs, prices correlate at r=0.988 with median divergence of -1c and 92% agreeing within 5c -- cross-platform arb is not viable at scale. |
| [§4.2](4_2_intra_event_parity.md) | Intra-Event Parity | 84% of 113K multi-outcome events deviate >5c from parity; 2-outcome events are underpriced (median -59c) while 6+ outcome events show escalating overround (~25-30c per additional contract). |
| [§4.3](4_3_conditional_correlated_arb.md) | Conditional/Correlated Arb | 48.4% of threshold events have monotonicity violations (higher threshold priced above lower), with crypto/equity families (KXBTCD, KXETHD) at 60-98% violation rates up to 98c. |

## Family 5: Agent / Address-Based (Polymarket)
| Report | Strategy | Key Finding |
|--------|----------|-------------|
| [§5.1](5_1_smart_money_following.md) | Smart Money Following | Top 1% of 474K Polymarket addresses earn +35.0 pp excess return (83.6% win rate at 48.6c avg price), while the bottom 90% lose -1.4 pp. |
| [§5.2](5_2_whale_detection.md) | Whale Detection | Whale trades (top 1% by size, >$1,231) earn +0.73 pp excess return vs -0.08 pp for the smallest trades -- a modest but statistically significant informational edge across 388M trades. |
| [§5.3](5_3_address_clustering.md) | Address Clustering | Market Makers (1.7% of addresses) are the only cluster with positive excess returns (+0.97 pp); Retail suffers -6.88 pp while Bot/HFT (1.0% of addresses, 71.7% of volume) operates near break-even. |
| [§5.4](5_4_contract_type_segmentation.md) | Contract Type Segmentation | CTF Exchange handles 264.5M trades with near-perfect calibration (ECE 0.005), while NegRisk (140.0M trades) has higher tail deviation (ECE 0.036) but a lower Brier score (0.099 vs 0.171). |

## Family 6: Behavioural / Sentiment
| Report | Strategy | Key Finding |
|--------|----------|-------------|
| [§6.1](6_1_contrarian_volume_spikes.md) | Contrarian Volume Spikes | Volume spike days (>3x average) show better calibration (1.68 pp MAE vs 2.58 pp normal), indicating spikes reflect informed trading rather than uninformed herding. |
| [§6.2](6_2_longshot_volume_share.md) | Longshot Volume Share | Longshot share (1-20c contracts) peaked at 6.3% of taker volume in Q4 2023 and declined to ~4.1% by late 2025 as total platform volume grew 1,500x. |
| [§6.3](6_3_yes_bias.md) | Yes Bias | YES takers outnumber NO takers 2.23:1 overall (46.8M vs 21.0M trades), peaking at ~4:1 in the 20-35c range, and overpay by an extra 0.52 pp. |

## Family 7: Composite / Multi-Signal
| Report | Strategy | Key Finding |
|--------|----------|-------------|
| [§7.1](7_1_filtered_market_making.md) | Filtered Market Making | Combining category selection (Entertainment/Sports/Crypto), NO-side tilt, and U.S.-hours timing yields an estimated 2-3x improvement over the +1.12 pp unfiltered maker baseline. |
| [§7.2](7_2_smart_money_informed_calibration.md) | Smart-Money-Informed Calibration | Smart money earns +35.0 pp buying at similar prices where naive longshot fading yields only -0.74 pp, implying selective longshot picking informed by wallet flow can vastly outperform blanket strategies. |
| [§7.3](7_3_cross_platform_relative_value.md) | Cross-Platform Relative Value | Cross-platform arb fails (1c median spread), but intra-platform parity violations (84% of events >5c off) and monotonicity violations (48.4%) combined with time-to-expiry convergence (3.2x decay) offer structural alpha. |

## Scripts
Standalone scripts are in `scripts/`. Run with: `uv run strategy_reports/scripts/X_Y_name.py`

## Figures
All figures are in `figures/` at 150 DPI.
