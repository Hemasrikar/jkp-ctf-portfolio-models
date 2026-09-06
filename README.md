# jkp-ctf-portfolio-models

### Sharpe results

> - Test period: From 1989/12/31 to 2023/12/31
> - volatility target: 10% for the scaled values.

| Model name | Sharpe value scaled | Sharpe value unscaled |
|---|---:|---:|
| [Coupled Factor Portfolio](./Coupled%20Factor%20Portfolio/) | 3.324 | 3.447 |
| [Volatility Targeted MLP](./Volatility%20Targeted%20MLP/) | 2.753 | 2.912 |
| [Beta-Neutral Cross-Sectional MLP](./Beta-Neutral%20Cross-Sectional%20MLP/beta_neutral_cross_sectional_mlp.py) | 2.541 | 2.614 |
| [Cross Sectional MLP](./Cross%20Sectional%20MLP/cross-sectional-mlp.py) | 2.484 | 2.553 |
| [MLP StockMixer Ensemble](./MLP%20StockMixer%20Ensemble/mlp-StockMixer-ensemble.py) | 2.405 | 2.438 |
| [State Conditional Cross Sectional MLP](./State%20Conditional%20Cross%20Sectional%20MLP/state-conditional-cossectional-mlp.py) | 2.373 | 2.336 |
| [StockMixer](./StockMixer/StockMixer.py) | 2.280 | 2.292 |
| [FT Crossectional Transformer](./FT%20Crossectional%20Transformer/ft-cross-portfolio.py) | 2.050 | 2.161 |
| [Cross Attention MLP](./Cross%20Attention%20MLP/cross-attention-mlp.py) | 1.940 | 1.935 |

---
As of now, the models provided above are ranked at decent positions on the [JKP Leaderboard](https://jkpfactors.com/ctf/leaderboard). There is a slight difference in the Sharpe ratios compared with the leaderboard, which is likely due to small floating-point differences from running the calculations on the GPU cluster.

---
## Important Findings

Findings are stated at the panel level where they hold across models

**The signal is close to fully extracted:** A variance decomposition of the
Coupled Factor Portfolio found an annualised alpha of 5.74% with a t-statistic
of 13.1 against eight standard characteristic factors, indicating statistically
significant selection skill. Only 20.6% of the portfolio's variance is explained
by common factors. The information ratio relative to these factors (2.42) is
below the realised Sharpe ratio (2.61), suggesting that the portfolio's
systematic exposures are, on net, rewarded rather than detrimental.

**Risk timing matters more than signal architecture:** For the Coupled Factor
Portfolio, scaling the portfolio to an ex-ante volatility target increased the
test Sharpe from 3.123 to 3.442 while leaving the underlying portfolio weights
unchanged. Across roughly thirty model specifications, no architectural change
altered the Sharpe by more than a few hundredths. Since the Sharpe ratio is
scale-invariant, the improvement therefore arises from varying portfolio
leverage through time rather than from changes in security selection.

**Conditional volatility is forecastable on this panel, whereas conditional
mean is not:** Volatility scaling consistently improved performance, while
attempts to time the conditional mean did not. More accurate volatility
forecasts did not translate into higher Sharpe ratios. Across five volatility
estimators, the QLIKE ranking was inversely related to the Sharpe ranking.
This suggests that the benefit of the overlay comes primarily from responding
to coarse movements in aggregate volatility rather than from precise
volatility forecasting.

**Expected return increases with volatility:** The optimal leverage exponent
is 0.75, below the value of 2 implied by a constant conditional mean. A separate
mean-timing regression also produced a positive log-variance coefficient
(t = 5.23). These two independent approaches therefore point in the same
direction, although the estimated effect is too small to motivate explicit
mean timing.

**Aggregate risk estimates help and per-stock estimates hurt the performance:** The successful
volatility overlay reduces the approximately 1,700-stock covariance structure
to a single aggregate risk estimate. Applying the same covariance information
at the individual-stock level consistently reduced performance for the Coupled
Factor Portfolio. Daily-estimated beta, idiosyncratic shrinkage, full-covariance
mean-variance optimisation, and MV−GMV reduced test Sharpe by 0.01, 0.59, 0.18,
and 0.82 respectively, and reduced pre-1990 Sharpe by 0.37, 0.96, 0.83, and
1.70. Realised volatility exceeded its forecast in every case, consistent with
optimisation being dominated by estimation error. With a cross-section
substantially wider than the estimation window, per-stock estimation error does
not average out sufficiently.

**Dimension reduction consistently loses information:** Compressing the 221
characteristics reduced performance across all tested approaches. Sliced
inverse regression to 40 directions achieved a Sharpe of 1.57. A learned
Grassmann subspace of 128 achieved 3.07. A fitted subspace of 48 achieved 2.97
with a network and 1.87 without one. For the Coupled Factor Portfolio,
performance falls away sharply on either side of `k_factors = 32`, with 16 and
64 factors producing Sharpe ratios of 1.54 and 1.96 respectively. Values
between 24 and 40 were not tested, so 32 is a working choice rather than a
fitted optimum.

**Cross-sectional coupling is the mechanism that matters:** Removing the
network, and therefore the pooled cross-sectional context, reduced Sharpe from
2.97 to 1.87 in the subspace model. This indicates that allowing a stock's
score or weight to depend on the composition of its contemporaneous cross
section is more important than any other individual architectural choice
tested.

**Self-supervised structure does not transfer directly to returns:** Four JEPA
designs, with each iteration addressing a previously identified limitation,
produced Sharpe ratios of 2.58, 2.90, 3.03, and 3.28. The rising sequence
reflects improvements to the underlying model rather than to the pretext task,
since the corresponding controls rose in step from 3.29 to 3.46 and each design
remained below its own control. The final design used genuinely distinct context
and target representations with horizon conditioning, suggesting that the
underperformance cannot readily be attributed to the previously identified
implementation limitations.

**Return magnitude contains information beyond cross-sectional ordering:**
Transformations that discard return magnitude progressively degraded
performance. Winsorisation was approximately neutral, standardisation reduced
Sharpe by 0.49, and replacing the target with its cross-sectional rank reduced
it by 2.20. Replacing the objective with a pairwise ranking loss reduced it by
2.18. The results therefore favour retaining the magnitude information in the
return target rather than reducing the objective to cross-sectional ordering.

**Pre-1990 selection is reliable for risk-side choices, but not capacity
choices.** The pre-1990 period correctly selected the covariance lookback that
performed best in the test period. In contrast, across three increases in model
width, pre-1990 performance increased monotonically while test performance
increased and then declined. Using the pre-1990 recommendation for
path-signature depth would therefore have reduced test Sharpe by 0.10.

**Several hyperparameters have little practical effect:** For the Coupled
Factor Portfolio, `gross_cap = 12` never binds, while `target_vol` is a pure
scale parameter, changing Sharpe by only 0.005. `moment_halflife` and
`cov_shrink` were also effectively flat across a fourfold range.