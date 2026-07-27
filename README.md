# jkp-ctf-portfolio-models

### Sharpe results

> - Test period: 1989/12/31
> - volatility target: 10% for the scaled values.

| Model name | Sharpe value scaled | Sharpe value unscaled |
|---|---|---|
| [Cross Attention MLP](./Cross%20Attention%20MLP/cross-attention-mlp.py) | 1.94 | 1.935 |
| [Cross Sectional MLP](./Cross%20Sectional%20MLP/cross-sectional-mlp.py) | 3.484 | 2.553 |
| [Beta-Neutral Cross-Sectional MLP](./Beta-Neutral%20Cross-Sectional%20MLP/beta_neutral_cross_sectional_mlp.py) | 2.541 | 2.614 |
| [FT Crossectional Transformer](./FT%20Crossectional%20Transformer/ft-cross-portfolio.py) | 2.05 | 2.161 |
| [MLP StockMixer Ensemble](./MLP%20StockMixer%20Ensemble/mlp-StockMixer-ensemble.py) | 2.405 | 2.438 |
| [State Conditional Cross Sectional MLP](./State%20Conditional%20Cross%20Sectional%20MLP/state-conditional-cossectional-mlp.py) | 2.373 | 2.336 |
| [StockMixer](./StockMixer/StockMixer.py) | 2.28 | 2.292 |