# jkp-ctf-portfolio-models

### Sharpe results

> - Test period: From 1989/12/31 to 2023/12/31
> - volatility target: 10% for the scaled values.

| Model name | Sharpe value scaled | Sharpe value unscaled | Doc |
|---|---:|---:| ---: |
| [Coupled Factor Portfolio](./Coupled%20Factor%20Portfolio/) | 3.324 | 3.447 | [pdf](./Coupled%20Factor%20Portfolio/coupled_factor_portfolio.pdf) |
| [Time Series Low Rank Portfolio](./Time%20Series%20Low%20Rank/)| 3.215 | 3.317 | [pdf](./Time%20Series%20Low%20Rank/time_series_low_rank.pdf) | 
| [Grassman Dimension Reduction Factor Portfolio](./Grassman%20Reduction/) | 2.842 | 3.046 | [pdf](./Grassman%20Reduction/grassmann_reduction.pdf) |
| [Volatility Targeted MLP](./Volatility%20Targeted%20MLP/) | 2.753 | 2.912 | [pdf](./Volatility%20Targeted%20MLP/volatility_targeted_mlp.pdf) |
| [Beta-Neutral Cross-Sectional MLP](./Beta-Neutral%20Cross-Sectional%20MLP/beta_neutral_cross_sectional_mlp.py) | 2.541 | 2.614 | [pdf](./Beta-Neutral%20Cross-Sectional%20MLP/beta_neutral_cross_sectional_mlp.pdf) |
| [Cross Sectional MLP](./Cross%20Sectional%20MLP/cross-sectional-mlp.py) | 2.484 | 2.553 | [pdf](./Cross%20Sectional%20MLP/cross_sectional_mlp.pdf) |
| [MLP StockMixer Ensemble](./MLP%20StockMixer%20Ensemble/mlp-StockMixer-ensemble.py) | 2.405 | 2.438 | [pdf](./MLP%20StockMixer%20Ensemble/mlp_stockmixer_ensemble.pdf) |
| [State Conditional Cross Sectional MLP](./State%20Conditional%20Cross%20Sectional%20MLP/state-conditional-cossectional-mlp.py) | 2.373 | 2.336 | [pdf](./State%20Conditional%20Cross%20Sectional%20MLP/state_conditional_cross_sectional_mlp.pdf) |
| [StockMixer](./StockMixer/StockMixer.py) | 2.280 | 2.292 | [pdf](./StockMixer/stockmixer.pdf) |
| [FT Crossectional Transformer](./FT%20Crossectional%20Transformer/ft-cross-portfolio.py) | 2.050 | 2.161 | [pdf](./FT%20Crossectional%20Transformer/ft_cross_sectional_transformer.pdf) |
| [Cross Attention MLP](./Cross%20Attention%20MLP/cross-attention-mlp.py) | 1.940 | 1.935 | [pdf](./Cross%20Attention%20MLP/cross_attention_mlp.pdf) |

---
As of now, the models provided above are ranked at decent positions on the [JKP Leaderboard](https://jkpfactors.com/ctf/leaderboard). There is a slight difference in the Sharpe ratios compared with the leaderboard, which is likely due to small floating-point differences from running the calculations on the GPU cluster.

---
## Data

The models are trained on the global factor data of Jensen, Kelly and Pedersen
(2023), which is constructed from CRSP and Compustat and distributed through
WRDS. In accordance with the WRDS subscriber agreement, this repository does
not contain raw data, processed feature files or trained model weights. Users
who wish to reproduce the results must obtain the data through their own WRDS
subscription and place it in the paths expected by each script.

Jensen, T. I., Kelly, B., and Pedersen, L. H. (2023). Is There a Replication
Crisis in Finance? *The Journal of Finance*, 78(5).

---
## License

The source code in this repository is released under the [MIT License](./LICENSE).
The accompanying PDF documents are released under the
[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
The license does not extend to the underlying data, which remains subject to
the terms of its original providers.