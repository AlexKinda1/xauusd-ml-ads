# Collecter macro + sentiment via Colab

La sandbox d'exécution de Claude n'a pas d'accès Internet. Cette étape se
déroule donc sur **Google Colab** (gratuit) ou n'importe quel environnement
Python avec accès réseau (machine locale, Kaggle, etc.).

## Procédure (5-10 min)

1. **Crée ta clé FRED gratuite** : https://fred.stlouisfed.org/docs/api/api_key.html
   (compte gratuit, clé en 30 secondes).

2. **Ouvre Colab** : https://colab.research.google.com →
   *File → Open notebook → GitHub* → coller `AlexKinda1/xauusd-ml-ads` →
   ouvrir `notebooks/00_colab_external_collection.ipynb`.

3. **Exécute les cellules dans l'ordre**. Le notebook te demandera :
   - Un **PAT GitHub** (optionnel — uniquement si tu veux que le notebook push
     les Parquets directement). Ne le partage pas par chat. Scope `repo` suffit.
   - Ta **clé FRED**.

4. **Résultat** : ~8 fichiers `macro_*.parquet` + ~2 fichiers `sentiment_*.parquet`
   dans `data/external/`, commités sur `claude/xau-usd-ml-prediction-DpLS9`.

## Et ensuite ?

Une fois les Parquets pushés, je relance dans la sandbox :

```bash
git pull origin claude/xau-usd-ml-prediction-DpLS9
PYTHONPATH=. python scripts/01_collect_all_data.py --skip-external   # re-aligne avec les nouveaux external/*.parquet
PYTHONPATH=. python scripts/02_build_features.py                     # rebuild features avec macro + sentiment
```

Les colonnes macro/sentiment seront automatiquement intégrées au `features_targets.parquet`
final puisque le pipeline (`src.data.align.load_external_parquets`) scanne `data/external/`.

## Notes

- **Google Trends** prend 3-5 min (rate-limit Google). Si pytrends échoue,
  le module continue sans bloquer (graceful failure documenté).
- **Fear & Greed** : source primaire = mirror CNN ; fallback automatique sur
  `alternative.me` (crypto F&G — proxy de risk-on/off).
- **Volumétrie** : ~5-10 MB total pour tous les Parquets. OK pour git en clair.
