# jkp-ctf-portfolio-models

### Sharpe results

> - Test period: From 1989/12/31 to 2023/12/31
> - volatility target: 10% for the scaled values.

| Model name | Sharpe value scaled | Sharpe value unscaled |
|---|---:|---:|
| [Monthly Volatility Targeted Coupled Factor Portfolio](./Monthly%20volatility%20targeted%20CFP/) | 3.324 | 3.447 |
| [IPCA](./IPCA/) - not my model | 3.299 | 3.350 |
| [Coupled Factor Portfolio](./Coupled%20Factor%20Portfolio/) | 2.969 | 3.112 |
| [Volatility Targeted MLP](./Volatility%20Targeted%20MLP/) | 2.753 | 2.912 |
| [Beta-Neutral Cross-Sectional MLP](./Beta-Neutral%20Cross-Sectional%20MLP/beta_neutral_cross_sectional_mlp.py) | 2.541 | 2.614 |
| [Cross Sectional MLP](./Cross%20Sectional%20MLP/cross-sectional-mlp.py) | 2.484 | 2.553 |
| [MLP StockMixer Ensemble](./MLP%20StockMixer%20Ensemble/mlp-StockMixer-ensemble.py) | 2.405 | 2.438 |
| [State Conditional Cross Sectional MLP](./State%20Conditional%20Cross%20Sectional%20MLP/state-conditional-cossectional-mlp.py) | 2.373 | 2.336 |
| [StockMixer](./StockMixer/StockMixer.py) | 2.280 | 2.292 |
| [FT Crossectional Transformer](./FT%20Crossectional%20Transformer/ft-cross-portfolio.py) | 2.050 | 2.161 |
| [Cross Attention MLP](./Cross%20Attention%20MLP/cross-attention-mlp.py) | 1.940 | 1.935 |

---
As of now, the models provided above are ranked 2nd, 3rd, 6th, and 7th, on the [JKP Leaderboard](https://jkpfactors.com/ctf/leaderboard). There is a slight difference in the Sharpe ratios compared with the leaderboard, which is likely due to small floating-point differences from running the calculations on the GPU cluster.

---
## Importation Findings

> cov_lookback_days controls how fast leverage responds. The optimal value found from the experimentation is 126 days