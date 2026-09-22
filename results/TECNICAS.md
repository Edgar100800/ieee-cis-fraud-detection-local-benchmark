# Técnicas para interiorizar — IEEE-CIS Fraud Detection
Análisis del código real de los 12 notebooks top (descargados en `codigos/`).

## 1. El dataset (por qué es difícil)
- 590,540 transacciones (train) × 394 columnas; 3.5% fraude → desbalance 1:27.
- **Split temporal**: train = 6 meses, test = el mes siguiente. Los modelos deben generalizar hacia el futuro, no memorizar el pasado.
- Familias de features: `card1-6` (tarjeta), `addr/dist` (geografía), `C1-14` (conteos), `D1-15` (deltas de tiempo), `M1-9` (matches), `V1-339` (features propietarias de Vesta, 43% NaN), identity (`id_01-38`, dispositivo) solo en 24% de filas.
- **Trampa de la métrica**: con AUC y 3.5% de fraude, un modelo que sobreajusta el pasado puede tener CV altísimo y fallar en test. Por eso todos los top usan validación temporal o GroupKFold por mes.

## 2. Baseline XGBoost (inversion y xhlulu) — AUC local 0.919/0.923
Qué hace: merge transaction+identity, `fillna(-999)`, LabelEncoder de categóricas, XGB depth 9 × 500.
Lecciones:
- Un XGB sin nada de ingeniería ya llega a ~0.92 local. Es el piso.
- **fill con -999 y `missing=-999`** funciona mejor que imputar con medias en GBMs: el árbol aprende que -999 es "desconocido" como categoría propia.
- xhlulu demuestra que el **mismo modelo en GPU (25s vs 112s)** permite iterar 4× más rápido. En XGBoost ≥2.0: `tree_method='hist', device='cuda'` (el viejo `gpu_hist` ya no existe).

## 3. Selección de features + count encoding (nroman) — AUC local 0.920
Qué hace:
- Lista de ~120 features "útiles" obtenida con **Recursive Feature Elimination** (su notebook hermano `nroman/recursive-feature-elimination`).
- **Count encoding**: reemplazar cada categoría por su frecuencia (`card1.map(value_counts())`) — la frecuencia de una tarjeta es señal de fraude (tarjetas con miles de transacciones = comprometidas).
- **FE temporal**: `day_of_week = floor((TransactionDT/86400 - 1) % 7)`, `hour = floor(TransactionDT/3600) % 24` (TransactionDT es segundos relativos).
- **Interacciones como string**: `card1__card5 = card1.astype(str)+'_'+card5.astype(str)` → luego LabelEncoder. Concatenar dos categóricas crea una "entidad" más fina.
- Decimal del monto: `(Amt - int(Amt)) * 1000` — detecta patrones de centavos típicos de bots.
- LightGBM con hiperparámetros de Bayesian optimization, TimeSeriesSplit, re-entrenamiento final con todo.
Lecciones: menos features bien elegidas ≈ mismo AUC con menos ruido; el count encoding es la forma más barata de "agregación" sin groupby.

## 4. ⭐ La técnica del 1er puesto (cdeotte "magic") — AUC local 0.9479 (+0.0104)
El notebook completo en 4 ideas:

### 4.1 Normalización de D-columns (alineación temporal)
```python
D15_normalizada = D15 - TransactionDT/(24*60*60)   # para D4,D5,D6,D8,D10,D11,D12,D13,D14,D15
```
Las columnas D son "días desde el último evento X" pero medidas respecto al momento de la transacción → crecen con el tiempo y el modelo confunde "mucho tiempo transcurrido" con "fraude". Al restar el tiempo absoluto, la feature queda **estable en el tiempo** y comparable entre train y test. (Deotte llegó a esto graficando D15 vs tiempo: el patrón se volvió una banda horizontal.)

### 4.2 Selección por time-consistency
Se eliminan features cuyo histograma cambia entre los primeros y últimos meses del train (ej. `C3, M5, id_08, id_33, card4...`). Razonamiento: si una feature no es estable en el tiempo, su relación con el fraude en test (futuro) será distinta → ruido.

### 4.3 UID mágico — reconstruir el cliente sin tener su ID
```python
uid = card1_addr1.astype(str) + '_' + floor(day - D1).astype(str)
```
- `card1+addr1` ≈ tarjeta; `day - D1` ≈ "fecha de apertura de la cuenta" (D1 es días desde la primera transacción). La combinación identifica al **cliente** aunque cambie de tarjeta/dispositivo.
- El UID **nunca entra al modelo**. Se usa solo para crear agregaciones por grupo y luego se borra:
  - `encode_FE`: frecuencia del grupo (¿cuántas transacciones ha hecho este cliente?).
  - `encode_AG`: media y std por grupo de `TransactionAmt, D4, D9, D10, D15, C1-C14, M1-M9` → la transacción se compara contra el historial de su propio cliente (desviación = anomalía).
  - `encode_AG2`: nunique por grupo (`P_emaildomain, dist1, id_02...`) → ¿cuántos valores distintos usa este cliente?
  - `outsider15 = |D1 - D15| > 3` → la dirección de envío es de "otro mundo" temporal.
- Efecto medido en este repo: **mismo modelo d12, AUC 0.9375 → 0.9479**. La ganancia es 100% representación de datos.
- Intuición (gráficos del notebook): un árbol solo con FeatureX acierta ~70% de un grupo de transacciones; con las medias por UID acierta 100% porque "la media del cliente" contextualiza cada transacción.

### 4.4 Validación correcta
Holdout 75/25 para iterar rápido + **GroupKFold con grupos = mes calendario** ( withholding un mes completo por fold): simula predecir el mes futuro, evita que las agregaciones por UID filtran información del mes de validación.

### 4.5 Post-procesado (XGB_96_PP)
Con una lista externa de UIDs "limpios", se **suaviza la probabilidad promediando dentro de cada UID** (transacciones del mismo cliente deberían tener predicciones parecidas). Subió el LB privado y les dio el 1er puesto en el ensemble con kyakovlev.

## 5. FE de Amount (kyakovlev GB-2, parte del stack ganador)
- `TransactionAmt_check = isin(Amt del otro set)` — flag de montos compartidos train/test.
- `log1p(TransactionAmt)` — el monto crudo tiene cola pesada; en log escala es útil.
- Agregaciones de `TransactionAmt` por card/cliente (misma idea encode_AG).
- Filosofía declarada: "deciéndole al modelo si confiar o no en estos valores" — transformar features para que la relación aprendida sea transferible al futuro.

## 6. EDA qué mirar (robikscube, jesucristo, kabure, artgor)
- Fraude vs tiempo: hay **picos cíclicos** de fraude (ataques por campañas); el fracaso de usar `TransactionDT` crudo.
- V-columns: bloques altamente correlacionados → Deotte las mapeó en su EDA de V's y eligió 1-2 por bloque (por eso el XGB_96 usa ~120 de las 339).
- ProductCD C y W concentran la mayoría del volumen y del fraude.
- email domain: `protonmail.com` tiene tasa de fraude altísima (feature de riesgo por dominio).
- artgor/kabure: pipelines completos EDA→FE→modelo→blend; artgor hace blend LGBM+CAT+XGB promediando ranks.

## 7. Desbalance (shahules)
- Con AUC como métrica NO hace falta SMOTE para GBMs (el árbol ya maneja desbalance); `scale_pos_weight`/`is_unbalance` son suficientes.
- SMOTE en 500k filas con 400 features es carísimo y puede crear fraudes sintéticos irreales. Útil conocerlo (Tomek links, undersampling, BalancedRandomForest) pero los top no lo usan en sus modelos finales.

## 8. Adversarial validation (tunguz) y memory reduction (mjbahmani, Deotte)
- Adversarial: entrenar un clasificador train-vs-test; si AUC alto, hay shift → las features culpables deben normalizarse (conecta con 4.1).
- `float32` en todo + `category` dtype: el dataset baja de 2.1GB a <1GB en RAM (en este repo aplicamos float32 en bench_04).
- RAPIDS cuDF (Deotte) hace todo el FE en GPU 15× más rápido, pero necesita >6GB VRAM → no ejecutable en RTX 2060 (verificado).

## 9. Reproducibilidad local (lo que corrimos)
| Benchmark | Comando | Resultado |
|---|---|---|
| 1 | `.venv/bin/python src/bench_01_inversion.py` | AUC 0.9191, 112s CPU |
| 2 | `.venv/bin/python src/bench_02_xhlulu.py` | AUC 0.9227, 25s GPU |
| 3 | `.venv/bin/python src/bench_03_nroman.py` | AUC 0.9200 (TS-CV 3), 462s CPU |
| 4 | `.venv/bin/python src/bench_04_cdeotte.py` | 0.9375 sin magic → **0.9479 con magic**, 192s GPU |

Adaptaciones necesarias en 2026 (documentadas en los scripts):
- XGBoost ≥2: `gpu_hist` → `tree_method='hist', device='cuda'`; `early_stopping_rounds` va en el constructor.
- LightGBM ≥4: `early_stopping_rounds`/`verbose_eval` → `callbacks=[lgb.early_stopping(), lgb.log_evaluation()]`.
- pandas 3.x: columnas string tienen dtype `str`, no `object` → usar `pd.api.types.is_numeric_dtype` en vez de `dtype=='object'`.

## 10. Qué probar después (orden de retorno esperado)
1. Añadir magic UID al LGBM de nroman (probable +0.01 también).
2. Ensemble XGB_96 + LGBM_magic (rank average) — el camino del 1er puesto real (0.9459 privado).
3. Post-procesado por UID sobre el ensemble.
4. Optimizar `max_depth=12, lr=0.02` con más folds (6 meses completos).
