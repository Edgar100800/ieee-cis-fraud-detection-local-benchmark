# IEEE-CIS Fraud Detection — Benchmark local de las mejores soluciones de Kaggle

Reproducción local y análisis comparativo de las mejores soluciones públicas de la
competencia [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)
(Kaggle, 2019, 6 355 equipos). **Todo el estudio se ejecutó en local, sin submissions.**

**Resultado central (ablación):** con el *mismo* modelo XGBoost d12, la técnica del
*magic UID* (reconstruir el pseudo-cliente `card1+addr1+floor(day−D1)` y agregar su
historial) eleva el AUC local de **0.9375 → 0.9479 (+0.0104)**. La ganancia proviene de la
representación de datos, no del algoritmo ni del hardware.

## Ranking local

| # | Método (notebook de origen) | AUC local | Tiempo | Validación |
|---|---|---:|---:|---|
| 1 | XGBoost d12 + **magic UID** (cdeotte, 1.er puesto) | **0.9479** | 192 s GPU | GroupKFold mensual ×3 |
| 2 | XGBoost d12 sin UID (cdeotte) | 0.9375 | 163 s GPU | GroupKFold mensual ×3 |
| 3 | LightGBM + RFE + count encoding (nroman) | 0.9200 | 462 s CPU | TimeSeriesSplit ×3 |
| 4 | XGBoost GPU (xhlulu) | 0.9227 | 25 s GPU | holdout temporal 75/25 |
| 5 | XGBoost CPU baseline (inversion) | 0.9191 | 112 s CPU | holdout temporal 75/25 |

Hardware: 16 núcleos CPU + NVIDIA RTX 2060 (6 GB). La GPU acelera 4.4× sin alterar el AUC.

## Técnicas clave documentadas

1. **Validación temporal honesta** — el split train/test de la competencia es temporal
   (train 6 meses → test el mes siguiente). Se usan holdout 75/25, `TimeSeriesSplit` y
   `GroupKFold` por mes calendario. KFold aleatorio infla el AUC.
2. **Magic UID** — el identificador de cliente no existe en los datos; se reconstruye,
   se usa solo como llave de agregación (~40 features de grupo: frecuencia, media/std,
   nunique) y se descarta antes de entrenar.
3. **Normalización de columnas D** — `D − TransactionDT/86400` elimina la deriva temporal
   y hace las features transferibles al futuro.
4. **Selección por consistencia temporal** — descartar variables cuyo histograma cambia
   dentro del train (serán ruido en test).
5. **Count encoding e interacciones** — frecuencia de tarjeta como señal de riesgo;
   `card1+card5`, `addr1+card1` como entidades más finas.
6. **Faltantes como categoría** — `fillna(-999)` + `missing=-999` supera la imputación en GBMs.

## Estructura

```
├── codigos/
│   ├── ejecutados/     4 notebooks reproducidos (.ipynb + .py extraído)
│   └── descartados/    8 notebooks analíticos (_LINKS.md con motivos)
├── src/                benchmarks reproducibles + descarga/exportación
├── results/            CSV de resultados, ranking y técnicas (TECNICAS.md)
├── informe_latex/      informe académico (28 págs., PDF compilado)
└── data/               (vacío en el repo: descargar de Kaggle, ver abajo)
```

## Reproducción

```bash
# 1. entorno
uv venv --python 3.12 .venv
uv pip install -r requirements.txt

# 2. datos (requiere cuenta Kaggle + aceptar reglas de la competencia)
kaggle competitions download -c ieee-fraud-detection -p data/raw
unzip 'data/raw/ieee-fraud-detection.zip' -d data/raw

# 3. benchmarks (en orden)
.venv/bin/python src/bench_01_inversion.py    # baseline CPU
.venv/bin/python src/bench_02_xhlulu.py       # baseline GPU
.venv/bin/python src/bench_03_nroman.py       # LightGBM
.venv/bin/python src/bench_04_cdeotte.py      # ablación magic UID

# 4. informe (figuras + PDF)
.venv/bin/python informe_latex/scripts/generar_graficos.py
cd informe_latex && tectonic main.tex
```

Los gráficos del informe se regeneran desde `data/raw` y `results/`: si cambias los
benchmarks, re-ejecuta el paso 4.

## Limitación conocida

Las agregaciones por UID se computan sobre todo el train (incluido el mes de validación,
sin usar sus etiquetas): protocolo transductivo, igual que el original de la competencia.
El siguiente hito del proyecto es la variante **estrictamente causal** (solo estadísticas
de transacciones anteriores a cada `TransactionDT`).

## Créditos y licencias

Código propio bajo [MIT](LICENSE). Los notebooks reproducidos mantienen su licencia
Apache 2.0 original y su autoría:

- Chris Deotte — [XGB Fraud with Magic](https://www.kaggle.com/code/cdeotte/xgb-fraud-with-magic-0-9600) (parte de la solución 1.er puesto)
- Nikolay Romanov — [LGB Single model](https://www.kaggle.com/code/nroman/lgb-single-model-lb-0-9419)
- xhlulu — [XGBoost with GPU](https://www.kaggle.com/code/xhlulu/ieee-fraud-xgboost-with-gpu-fit-in-40s)
- inversion — [IEEE Simple XGBoost](https://www.kaggle.com/code/inversion/ieee-simple-xgboost)
- Índice completo de los 12 notebooks en `codigos/descartados/_LINKS.md` y `referencias.bib`.
