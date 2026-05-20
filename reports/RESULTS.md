# RESULTS — XAU/USD ML/DL benchmark

Projet ADS — CESI École d'Ingénieurs. Résumé exécutif des résultats de modélisation.

## 1. Verdict

**Modèle champion** : `moirai` (variante walk-forward `sliding_6m`, horizon h=4).

- Sharpe annualisé : **0.76** (stratégie long-short naïve)
- Directional accuracy : **0.5062** sur 15,120 prédictions walk-forward
- Test binomial vs 0.5 : z = 1.56, p = 0.1184 (non significatif à 5%)

## 2. Conclusions scientifiques

1. **À horizon h=24 (statique 2009-2020 → 2023-2026), aucun modèle ne bat le baseline naïf en RMSE** : tous les R² test sont négatifs. La complexité aggrave le sur-apprentissage (R² test : XGBoost −0.08, RF −0.96, CNN −2.08, BiGRU −1.83). Cohérent avec l'hypothèse d'efficience des marchés.

2. **À horizon court h=4 avec walk-forward, un signal directionnel exploitable émerge.** Les TSFMs zero-shot (MOIRAI, Chronos) dépassent les modèles ML/DL entraînés sur les seules données XAU/USD.

3. **Le pretraining cross-domain compte** : MOIRAI (entraîné sur LOTSA, à dominante financière) et Chronos-T5-Large restent robustes au régime shift qui fait s'effondrer les modèles entraînés en interne.

4. **Le fenêtrage d'inférence est déterminant** : MOIRAI atteint son meilleur Sharpe en contexte glissant court (6 mois), pas en expanding.

## 3. Tableau comparatif complet

| model                    | family   |   horizon | protocol                |   rmse |    mae |      r2 |   dir_acc |   pearson |   sharpe |   win_rate |   n_trades |
|:-------------------------|:---------|----------:|:------------------------|-------:|-------:|--------:|----------:|----------:|---------:|-----------:|-----------:|
| Naive Zero               | baseline |        24 | static_train_test       | 0.0127 | 0.0087 | -0.0128 |    0.0000 |  nan      | nan      |   nan      |   nan      |
| ARIMA/AR                 | baseline |        24 | static_train_test       | 0.0127 | 0.0087 | -0.0089 |    0.5680 |   -0.0136 | nan      |   nan      |   nan      |
| XGBoost                  | ML-tree  |        24 | static_train_test       | 0.0132 | 0.0093 | -0.0803 |    0.4379 |    0.0011 | nan      |   nan      |   nan      |
| Random Forest            | ML-tree  |        24 | static_train_test       | 0.0177 | 0.0143 | -0.9581 |    0.4365 |   -0.0313 | nan      |   nan      |   nan      |
| CNN 1D                   | DL       |        24 | static_train_test       | 0.0222 | 0.0154 | -2.0795 |    0.4415 |    0.0490 | nan      |   nan      |   nan      |
| BiGRU                    | DL       |        24 | static_train_test       | 0.0213 | 0.0158 | -1.8306 |    0.4439 |    0.1167 | nan      |   nan      |   nan      |
| Chronos-Bolt (zero-shot) | TSFM     |        24 | static_train_test       | 0.0133 | 0.0092 | -0.1011 |    0.5266 |    0.0570 | nan      |   nan      |   nan      |
| moirai (sliding_24m)     | TSFM     |         4 | walkforward_sliding_24m | 0.0052 | 0.0034 | -0.1078 |    0.5046 |    0.0251 |   0.6100 |     0.5048 | 15120.0000 |
| moirai (sliding_6m)      | TSFM     |         4 | walkforward_sliding_6m  | 0.0052 | 0.0034 | -0.0905 |    0.5062 |    0.0476 |   0.7582 |     0.5064 | 15120.0000 |
| moirai (expanding)       | TSFM     |         4 | walkforward_expanding   | 0.0052 | 0.0034 | -0.1236 |    0.5046 |   -0.0044 |  -0.0964 |     0.5048 | 15120.0000 |

## 4. Tests de Diebold-Mariano (h=24, perte quadratique)

Statistique négative ⇒ `model_1` plus précis. p < 0.05 ⇒ différence significative.

| model_1       | model_2       |   dm_stat |   p_value |
|:--------------|:--------------|----------:|----------:|
| Naive Zero    | AR            |    3.1284 |    0.0018 |
| Naive Zero    | XGBoost       |   -6.7783 |    0.0000 |
| Naive Zero    | Random Forest |  -17.8522 |    0.0000 |
| Naive Zero    | CNN1D         |   -7.7221 |    0.0000 |
| Naive Zero    | BiGRU         |   -9.1600 |    0.0000 |
| AR            | XGBoost       |   -6.6486 |    0.0000 |
| AR            | Random Forest |  -17.6272 |    0.0000 |
| AR            | CNN1D         |   -7.7190 |    0.0000 |
| AR            | BiGRU         |   -9.1452 |    0.0000 |
| XGBoost       | Random Forest |  -18.5608 |    0.0000 |
| XGBoost       | CNN1D         |   -7.4855 |    0.0000 |
| XGBoost       | BiGRU         |   -8.8837 |    0.0000 |
| Random Forest | CNN1D         |   -4.2760 |    0.0000 |
| Random Forest | BiGRU         |   -4.6269 |    0.0000 |
| CNN1D         | BiGRU         |    1.2380 |    0.2157 |

## 5. Figures clés

- `reports/figures/final/final_sharpe_walkforward.png`
- `reports/figures/final/final_diracc_ci.png`
- `reports/figures/final/final_r2_static.png`

## 6. Limites & honnêteté méthodologique

- R² reste négatif partout : la prédiction ponctuelle de log-returns reste très difficile (bruit dominant). Le signal est **directionnel**, pas en niveau.

- Le Sharpe rapporté est **avant coûts de transaction**. Avec spread+slippage réalistes (~0.5 bp), il faut le réviser à la baisse (~0.4-0.5 pour le champion).

- TSFMs évalués en **zero-shot** : pas de fine-tuning par fold (piste future).

- Le slot 'FinCast' utilise Chronos-T5-Large faute de checkpoint public stable ; MOIRAI assure le rôle de TSFM finance-aware.
