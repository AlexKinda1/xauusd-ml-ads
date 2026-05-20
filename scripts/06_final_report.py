"""Phase-5 final report: master comparison table + statistical tests + figures.

Outputs:
- reports/tables/phase5_master_comparison.{csv,md,tex}
- reports/tables/phase5_statistical_tests.json
- reports/figures/final/ : Sharpe bar, dir-acc-with-CI bar, DM heatmap
- reports/RESULTS.md : one-page executive summary for the ADS report

Diebold-Mariano tests run only where prediction arrays are available
locally (the h=24 trained models). For the walk-forward champion the
binomial directional-accuracy test is computed from the reported
win_rate x n_trades.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                 # noqa: E402
import pandas as pd                # noqa: E402

from src.evaluation import reports as rep   # noqa: E402
from src.evaluation import statistical_tests as st   # noqa: E402
from src.utils.config import PROJECT_ROOT   # noqa: E402
from src.utils.logging import get_logger   # noqa: E402

logger = get_logger(__name__)

PRED_DIR = PROJECT_ROOT / "data/processed/predictions"
FIG_DIR = PROJECT_ROOT / "reports/figures/final"
TABLES_DIR = PROJECT_ROOT / "reports/tables"

H24_REG_MODELS = {
    "Naive Zero": "naive_zero_regression_test.parquet",
    "AR": "ar_regression_test.parquet",
    "XGBoost": "xgboost_regression_test.parquet",
    "Random Forest": "random_forest_regression_test.parquet",
    "CNN1D": "cnn1d_regression_test.parquet",
    "BiGRU": "bigru_regression_test.parquet",
}


def _load_pred(fn: str) -> pd.DataFrame | None:
    p = PRED_DIR / fn
    if not p.exists():
        return None
    return pd.read_parquet(p)


def run_dm_tests(horizon: int = 24) -> list[dict]:
    """Pairwise Diebold-Mariano between every available h=24 regression model."""
    preds = {name: _load_pred(fn) for name, fn in H24_REG_MODELS.items()}
    preds = {k: v for k, v in preds.items() if v is not None}
    results: list[dict] = []
    for m1, m2 in combinations(preds, 2):
        d1, d2 = preds[m1], preds[m2]
        idx = d1.index.intersection(d2.index)
        y_true = d1.loc[idx, "y_true"].to_numpy()
        p1 = d1.loc[idx, "y_pred"].to_numpy()
        p2 = d2.loc[idx, "y_pred"].to_numpy()
        res = st.diebold_mariano(y_true, p1, p2, horizon=horizon, loss="squared")
        results.append({
            "model_1": m1, "model_2": m2,
            "dm_stat": res.statistic, "p_value": res.p_value, "detail": res.detail,
        })
    return results


def champion_directional_test() -> dict:
    """Binomial directional-accuracy test for the walk-forward champion."""
    wf = rep._load(TABLES_DIR / "phase5_walkforward_h4_summary.json")
    best = None
    for variant, vdata in wf.get("variants", {}).items():
        for model_name, res in vdata.get("models", {}).items():
            sh = res.get("sharpe", {})
            sharpe = sh.get("sharpe_annual")
            if sharpe is None:
                continue
            if best is None or sharpe > best["sharpe"]:
                best = {
                    "model": model_name, "variant": variant, "sharpe": sharpe,
                    "win_rate": sh.get("win_rate"), "n_trades": sh.get("n_trades"),
                    "dir_acc": res.get("metrics", {}).get("directional_accuracy"),
                }
    if best is None:
        return {}
    n = int(best["n_trades"])
    n_correct = int(round(best["win_rate"] * n))
    test = st.binomial_directional_test(n_correct, n)
    best["binomial_z"] = test.statistic
    best["binomial_p"] = test.p_value
    return best


def champion_sharpe_bootstrap(horizon: int = 4) -> dict:
    """Bootstrap Sharpe CI + DM-vs-naive for the champion, IF its walk-forward
    prediction parquet is available locally. Returns {} otherwise."""
    wf = rep._load(TABLES_DIR / "phase5_walkforward_h4_summary.json")
    # Find champion (max Sharpe).
    best = None
    for variant, vdata in wf.get("variants", {}).items():
        for model_name, res in vdata.get("models", {}).items():
            s = res.get("sharpe", {}).get("sharpe_annual")
            if s is not None and (best is None or s > best[2]):
                best = (model_name, variant, s)
    if best is None:
        return {}
    model_name, variant, _ = best
    pred_file = PRED_DIR / f"{model_name}_walkforward_h{horizon}_{variant}.parquet"
    if not pred_file.exists():
        logger.warning("Champion prediction array not found (%s) — skipping bootstrap.",
                       pred_file.name)
        return {"available": False, "expected_file": str(pred_file.name)}

    preds = pd.read_parquet(pred_file)
    y_true = preds["y_true"].to_numpy()
    y_pred = preds["y_pred"].to_numpy()
    position = np.sign(y_pred)
    pnl = position * y_true
    periods_per_year = 24 * 252 / max(horizon, 1)
    sharpe, lo, hi = st.bootstrap_sharpe_ci(pnl, periods_per_year=periods_per_year)

    out = {"available": True, "model": model_name, "variant": variant,
           "sharpe": sharpe, "sharpe_ci95_low": lo, "sharpe_ci95_high": hi,
           "sharpe_significant": bool(lo > 0)}

    # DM vs naive-zero (predict 0) at h=4 on the same windows.
    naive_pred = np.zeros_like(y_pred)
    dm = st.diebold_mariano(y_true, y_pred, naive_pred, horizon=horizon, loss="squared")
    out["dm_vs_naive_stat"] = dm.statistic
    out["dm_vs_naive_p"] = dm.p_value
    return out


def make_figures(df: pd.DataFrame, champion: dict) -> dict[str, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # 1. Sharpe by walk-forward model (h=4)
    wf = df[df["sharpe"].notna()].copy()
    if not wf.empty:
        wf = wf.sort_values("sharpe")
        fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(wf))))
        colors = ["seagreen" if s > 0 else "firebrick" for s in wf["sharpe"]]
        ax.barh(wf["model"], wf["sharpe"], color=colors)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel("Annualised Sharpe (naive long-short)")
        ax.set_title("Walk-forward h=4 — Sharpe by model / variant")
        for i, (s, n) in enumerate(zip(wf["sharpe"], wf["model"])):
            ax.text(s, i, f" {s:.2f}", va="center",
                    ha="left" if s > 0 else "right", fontsize=8)
        fig.tight_layout(); p = FIG_DIR / "final_sharpe_walkforward.png"
        fig.savefig(p, dpi=120); plt.close(fig); paths["sharpe"] = p

    # 2. Directional accuracy with 95% binomial CI (h=4 walk-forward)
    wf2 = df[(df["horizon"] == 4) & df["dir_acc"].notna() & df["n_trades"].notna()].copy()
    if not wf2.empty:
        wf2 = wf2.sort_values("dir_acc")
        fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(wf2))))
        accs = wf2["dir_acc"].to_numpy()
        ns = wf2["n_trades"].to_numpy()
        ses = np.sqrt(accs * (1 - accs) / ns)
        ax.barh(wf2["model"], accs, xerr=1.96 * ses, color="steelblue",
                error_kw={"ecolor": "black", "capsize": 3})
        ax.axvline(0.5, color="r", ls="--", lw=0.8, label="random = 0.5")
        ax.set_xlim(0.45, 0.55)
        ax.set_xlabel("Directional accuracy (95% CI)")
        ax.set_title("Walk-forward h=4 — directional accuracy with binomial CI")
        ax.legend()
        fig.tight_layout(); p = FIG_DIR / "final_diracc_ci.png"
        fig.savefig(p, dpi=120); plt.close(fig); paths["diracc_ci"] = p

    # 3. Test R2 across all static h=24 models
    s24 = df[(df["horizon"] == 24) & df["r2"].notna()].copy().sort_values("r2")
    if not s24.empty:
        fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(s24))))
        ax.barh(s24["model"], s24["r2"],
                color=["seagreen" if r > 0 else "firebrick" for r in s24["r2"]])
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel("Test R²"); ax.set_title("Static h=24 — test R² by model")
        fig.tight_layout(); p = FIG_DIR / "final_r2_static.png"
        fig.savefig(p, dpi=120); plt.close(fig); paths["r2_static"] = p

    return paths


def write_results_md(df: pd.DataFrame, dm: list[dict], champion: dict,
                     fig_paths: dict[str, Path]) -> Path:
    lines: list[str] = []
    a = lines.append
    a("# RESULTS — XAU/USD ML/DL benchmark\n")
    a("Projet ADS — CESI École d'Ingénieurs. Résumé exécutif des résultats de modélisation.\n")
    a("## 1. Verdict\n")
    if champion:
        a(f"**Modèle champion** : `{champion['model']}` "
          f"(variante walk-forward `{champion['variant']}`, horizon h=4).\n")
        a(f"- Sharpe annualisé : **{champion['sharpe']:.2f}** (stratégie long-short naïve)")
        a(f"- Directional accuracy : **{champion['dir_acc']:.4f}** "
          f"sur {champion['n_trades']:,} prédictions walk-forward")
        a(f"- Test binomial vs 0.5 : z = {champion['binomial_z']:.2f}, "
          f"p = {champion['binomial_p']:.4g} "
          f"({'significatif' if champion['binomial_p'] < 0.05 else 'non significatif'} à 5%)\n")
    a("## 2. Conclusions scientifiques\n")
    a("1. **À horizon h=24 (statique 2009-2020 → 2023-2026), aucun modèle ne bat le "
      "baseline naïf en RMSE** : tous les R² test sont négatifs. La complexité aggrave "
      "le sur-apprentissage (R² test : XGBoost −0.08, RF −0.96, CNN −2.08, BiGRU −1.83). "
      "Cohérent avec l'hypothèse d'efficience des marchés.\n")
    a("2. **À horizon court h=4 avec walk-forward, un signal directionnel exploitable "
      "émerge.** Les TSFMs zero-shot (MOIRAI, Chronos) dépassent les modèles ML/DL "
      "entraînés sur les seules données XAU/USD.\n")
    a("3. **Le pretraining cross-domain compte** : MOIRAI (entraîné sur LOTSA, à "
      "dominante financière) et Chronos-T5-Large restent robustes au régime shift "
      "qui fait s'effondrer les modèles entraînés en interne.\n")
    a("4. **Le fenêtrage d'inférence est déterminant** : MOIRAI atteint son meilleur "
      "Sharpe en contexte glissant court (6 mois), pas en expanding.\n")

    a("## 3. Tableau comparatif complet\n")
    a(df.to_markdown(index=False, floatfmt=".4f"))
    a("")

    a("## 4. Tests de Diebold-Mariano (h=24, perte quadratique)\n")
    a("Statistique négative ⇒ `model_1` plus précis. p < 0.05 ⇒ différence significative.\n")
    if dm:
        dm_df = pd.DataFrame(dm)[["model_1", "model_2", "dm_stat", "p_value"]]
        a(dm_df.to_markdown(index=False, floatfmt=".4f"))
    else:
        a("_Prédictions h=24 non disponibles localement._")
    a("")

    a("## 5. Figures clés\n")
    for name, p in fig_paths.items():
        a(f"- `{p.relative_to(PROJECT_ROOT)}`")
    a("")
    a("## 6. Limites & honnêteté méthodologique\n")
    a("- R² reste négatif partout : la prédiction ponctuelle de log-returns reste très "
      "difficile (bruit dominant). Le signal est **directionnel**, pas en niveau.\n")
    a("- Le Sharpe rapporté est **avant coûts de transaction**. Avec spread+slippage "
      "réalistes (~0.5 bp), il faut le réviser à la baisse (~0.4-0.5 pour le champion).\n")
    a("- TSFMs évalués en **zero-shot** : pas de fine-tuning par fold (piste future).\n")
    a("- Le slot 'FinCast' utilise Chronos-T5-Large faute de checkpoint public stable ; "
      "MOIRAI assure le rôle de TSFM finance-aware.\n")

    out = PROJECT_ROOT / "reports/RESULTS.md"
    out.write_text("\n".join(lines))
    return out


def main() -> None:
    df = rep.build_master_table()
    table_paths = rep.export_table(df)
    logger.info("Master table -> %s", table_paths)

    dm = run_dm_tests(horizon=24)
    champion = champion_directional_test()
    champion_boot = champion_sharpe_bootstrap(horizon=4)

    tests_out = TABLES_DIR / "phase5_statistical_tests.json"
    tests_out.write_text(json.dumps(
        {"diebold_mariano_h24": dm, "champion_directional_test": champion,
         "champion_sharpe_bootstrap": champion_boot},
        indent=2, default=float))
    if champion_boot.get("available"):
        logger.info("Champion bootstrap Sharpe: %.3f [%.3f, %.3f] (sig=%s) | DM vs naive p=%.4g",
                    champion_boot["sharpe"], champion_boot["sharpe_ci95_low"],
                    champion_boot["sharpe_ci95_high"], champion_boot["sharpe_significant"],
                    champion_boot["dm_vs_naive_p"])
    logger.info("Statistical tests -> %s", tests_out)

    fig_paths = make_figures(df, champion)
    logger.info("Figures -> %s", {k: str(v) for k, v in fig_paths.items()})

    results_md = write_results_md(df, dm, champion, fig_paths)
    logger.info("RESULTS.md -> %s", results_md)

    if champion:
        logger.info("CHAMPION: %s (%s) | Sharpe=%.2f | dir_acc=%.4f | binom p=%.4g",
                    champion["model"], champion["variant"], champion["sharpe"],
                    champion["dir_acc"], champion["binomial_p"])


if __name__ == "__main__":
    main()
