# Resumen de técnicas por notebook

Inventario de los 5 notebooks ejecutados (`codigos/ejecutados/`), todos con outputs
visibles tras la última corrida local (datos en `data/raw/`).

## 01_xgboost_baseline_cpu
- Protocolo: holdout temporal 75/25 (implícito: fit directo + predicción de test)
- Técnica: baseline XGBoost CPU (`n_estimators=500`, `max_depth=9`), merge
  transaction+identity, `fillna(-999)`, LabelEncoding de categóricas (incluye el
  fix `id-` → `id_` del CSV de test).
- Semilla: sin semilla explícita (XGBoost usa su default `seed=0`).

## 02_xgboost_gpu
- Protocolo: fit directo en GPU (~40 s con RTX 2060).
- Técnica: XGBoost `device='cuda'` (antes `gpu_hist`), mismo preprocesado que 01,
  `random_state=2019` fijado para reproducibilidad.
- Semilla: `random_state=2019`.

## 03_lightgbm_encoding
- Protocolo: `TimeSeriesSplit(n_splits=5)` con early stopping (500) vía callbacks
  de LightGBM 4.x.
- Técnica: count encoding (`card1_count_full`, `id_*_count_full/_dist`),
  features de día/hora, interacciones `id_02__id_20`, `card1__card5`,
  `addr1__card1`, etc., selección de ~120 features útiles, imports de
  importancia por fold.
- Resultado de la corrida: AUC medio 0.9240 (5 folds).
- Semilla: `random_state=47`, `bagging_seed=11`.

## 04_xgboost_magic_uid
- Protocolo: holdout temporal 75/25 + GroupKFold mensual ×6 con early stopping
  (XGBoost 3.x: `early_stopping_rounds` en el constructor).
- Técnica: ablación magic UID — columnas D normalizadas, selección de columnas V
  por correlación, FE de agregación y frecuencia; UID = `card1+addr1+floor(day−D1)`
  + ~47 agregaciones de grupo; el UID se descarta antes de entrenar.
- Resultado de la corrida: sin UID holdout AUC 0.9365 / OOF 0.9399; con magic UID
  0.9450 / 0.9534.
- Semilla: sin semilla explícita (split determinístico por grupos de mes).

## 05_sistema_integrado (propio)
- Protocolo: holdout temporal 75/25 + OOF GroupKFold mensual ×3; ablación sin/con UID.
- Técnica: integración de lo mejor de cada notebook — D normalizadas (04), magic UID
  + agregaciones (04), count encoding e interacciones `card1×card5`/`addr1×card1` (03),
  faltantes como categoría (01/02) — sobre XGBoost d12 GPU.
- Resultado de la corrida: sin UID holdout 0.9340 / OOF 0.9347; con magic UID
  0.9456 / 0.9509 → ganancia de representación +0.0116 / +0.0162.
- Semilla: `SEED=42` (global, pasada a XGBoost vía `random_state`).
