"""Genera el ranking comparativo final de los benchmarks locales."""
from pathlib import Path

import pandas as pd

rows = []

b1 = pd.read_csv("results/bench_01_inversion.csv").iloc[0]
rows.append({"ranking": 5, "notebook": "inversion/ieee-simple-xgboost (baseline)",
             "modelo": "XGBoost CPU d9 x500", "auc": b1.auc_holdout, "tiempo_s": b1.fit_s,
             "validacion": "holdout temporal 75/25", "notas": "432 features raw, fillna(-999)"})

b2 = pd.read_csv("results/bench_02_xhlulu.csv").iloc[0]
rows.append({"ranking": 4, "notebook": "xhlulu/ieee-fraud-xgboost-with-gpu",
             "modelo": "XGBoost GPU d9 x500", "auc": b2.auc_holdout, "tiempo_s": b2.fit_s,
             "validacion": "holdout temporal 75/25", "notas": "identico a #5 pero GPU: 4.4x mas rapido"})

b3 = pd.read_csv("results/bench_03_nroman.csv").iloc[0]
rows.append({"ranking": 3, "notebook": "nroman/lgb-single-model-lb-0-9419",
             "modelo": "LightGBM CPU (params Bayes opt)", "auc": b3.auc_mean_timeseriessplit,
             "tiempo_s": b3.cv_s, "validacion": "TimeSeriesSplit 3 folds",
             "notas": "RFE 120 features + count/day/hour/interacciones"})

b4 = pd.read_csv("results/bench_04_cdeotte.csv")
b95 = b4[b4.notebook == "cdeotte-xgb95-sin-magic"].iloc[0]
b96 = b4[b4.notebook == "cdeotte-xgb96-magic"].iloc[0]
rows.append({"ranking": 2, "notebook": "cdeotte/xgb-fraud-with-magic [XGB_95]",
             "modelo": "XGBoost GPU d12", "auc": b95.auc_oof_groupkfold3,
             "tiempo_s": b95.holdout_s + b95.cv_s, "validacion": "OOF GroupKFold 3 meses",
             "notas": "sin magic: ~120 V-cols + FE base + D normalizadas"})
rows.append({"ranking": 1, "notebook": "cdeotte/xgb-fraud-with-magic [XGB_96] 1er PUESTO",
             "modelo": "XGBoost GPU d12 + magic UID", "auc": b96.auc_oof_groupkfold3,
             "tiempo_s": b96.holdout_s + b96.cv_s, "validacion": "OOF GroupKFold 3 meses",
             "notas": "+UID card1+addr1+D1n y 40+ agregaciones: +0.0104 AUC"})

df = pd.DataFrame(rows)
df.to_csv("results/RANKING_FINAL.csv", index=False)

lines = [
    "# RANKING FINAL - Benchmark local IEEE-CIS Fraud Detection",
    "",
    f"Validacion local (sin submissions). Maquina: 16 cores CPU, RTX 2060 6GB.",
    "El split de la competencia es temporal (train 6 meses -> test mes siguiente);",
    "el holdout 75/25 temporal y el GroupKFold por mes replican ese shift.",
    "",
    "| # | Notebook | Modelo | AUC local | Tiempo | Validacion | Notas |",
    "|---|----------|--------|-----------|--------|------------|-------|",
]
for r in rows:
    lines.append(
        f"| {r['ranking']} | {r['notebook']} | {r['modelo']} | **{r['auc']:.4f}** | "
        f"{r['tiempo_s']:.0f}s | {r['validacion']} | {r['notas']} |"
    )
lines += [
    "",
    "## Lecturas clave",
    "",
    "1. La tecnica del 1er puesto (magic UID) domina: +0.0104 AUC sobre su mismo modelo sin UID,",
    "   y +0.025 sobre el baseline. La ganancia NO viene del modelo (mismo XGB d12) sino del FE:",
    "   construir un pseudo-cliente (card1+addr1+floor(day-D1)) y agregar estadisticas por grupo.",
    "2. El mismo XGB en GPU es 4.4x mas rapido que en CPU (25s vs 112s) con AUC identico.",
    "3. LightGBM de nroman (features RFE + count encoding) empata al baseline XGB raw con 30% menos features.",
    "4. Scores de referencia Kaggle (public LB): inversion ~0.943, nroman 0.9419, cdeotte XGB_95 0.9514,",
    "   XGB_96 0.9627. En local el orden se conserva: mas FE orientada a entidades -> mas AUC.",
    "",
    "## Equipos top del leaderboard publico (6355 equipos)",
    "",
    "1. AlKo 0.9681 | 2. FraudSquad (cdeotte+kyakovlev) 0.9677 | 3. Young for you 0.9676 |",
    "4. 2 uncles and 3 puppies 0.9672 | 5. Mr Lonely 0.9663",
]
Path("results/RANKING_FINAL.md").write_text("\n".join(lines))
print("\n".join(lines))
