# RANKING FINAL - Benchmark local IEEE-CIS Fraud Detection

Validacion local (sin submissions). Maquina: 16 cores CPU, RTX 2060 6GB.
El split de la competencia es temporal (train 6 meses -> test mes siguiente);
el holdout 75/25 temporal y el GroupKFold por mes replican ese shift.

| # | Notebook | Modelo | AUC local | Tiempo | Validacion | Notas |
|---|----------|--------|-----------|--------|------------|-------|
| 5 | inversion/ieee-simple-xgboost (baseline) | XGBoost CPU d9 x500 | **0.9191** | 112s | holdout temporal 75/25 | 432 features raw, fillna(-999) |
| 4 | xhlulu/ieee-fraud-xgboost-with-gpu | XGBoost GPU d9 x500 | **0.9227** | 25s | holdout temporal 75/25 | identico a #5 pero GPU: 4.4x mas rapido |
| 3 | nroman/lgb-single-model-lb-0-9419 | LightGBM CPU (params Bayes opt) | **0.9200** | 462s | TimeSeriesSplit 3 folds | RFE 120 features + count/day/hour/interacciones |
| 2 | cdeotte/xgb-fraud-with-magic [XGB_95] | XGBoost GPU d12 | **0.9375** | 163s | OOF GroupKFold 3 meses | sin magic: ~120 V-cols + FE base + D normalizadas |
| 1 | cdeotte/xgb-fraud-with-magic [XGB_96] 1er PUESTO | XGBoost GPU d12 + magic UID | **0.9479** | 192s | OOF GroupKFold 3 meses | +UID card1+addr1+D1n y 40+ agregaciones: +0.0104 AUC |

## Lecturas clave

1. La tecnica del 1er puesto (magic UID) domina: +0.0104 AUC sobre su mismo modelo sin UID,
   y +0.025 sobre el baseline. La ganancia NO viene del modelo (mismo XGB d12) sino del FE:
   construir un pseudo-cliente (card1+addr1+floor(day-D1)) y agregar estadisticas por grupo.
2. El mismo XGB en GPU es 4.4x mas rapido que en CPU (25s vs 112s) con AUC identico.
3. LightGBM de nroman (features RFE + count encoding) empata al baseline XGB raw con 30% menos features.
4. Scores de referencia Kaggle (public LB): inversion ~0.943, nroman 0.9419, cdeotte XGB_95 0.9514,
   XGB_96 0.9627. En local el orden se conserva: mas FE orientada a entidades -> mas AUC.

## Equipos top del leaderboard publico (6355 equipos)

1. AlKo 0.9681 | 2. FraudSquad (cdeotte+kyakovlev) 0.9677 | 3. Young for you 0.9676 |
4. 2 uncles and 3 puppies 0.9672 | 5. Mr Lonely 0.9663