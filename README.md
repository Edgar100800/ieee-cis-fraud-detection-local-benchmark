# IEEE-CIS Fraud Detection — Sistema adaptativo y benchmark temporal

Reproducción local y análisis comparativo de las mejores soluciones públicas de la
competencia [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)
(Kaggle, 2019, 6 355 equipos). **Todo el estudio se ejecutó en local, sin submissions.**

**Resultado central (ablación):** con el *mismo* modelo XGBoost d12, la técnica del
*magic UID* (reconstruir el pseudo-cliente `card1+addr1+floor(day−D1)` y agregar su
historial) eleva el AUC local de **0.9375 → 0.9479 (+0.0104)**. La ganancia proviene de la
representación de datos, no del algoritmo ni del hardware.

**Extensión adaptativa:** sobre la línea base se implementó una evaluación por bloques futuros
con features UID causales, ventanas expansivas y deslizantes de 30/60/90 días, PSI para monitoreo
de drift y métricas de decisión. En la corrida completa, la ventana expansiva obtuvo AUC promedio
0.9163 y la ventana de 30 días 0.9039; el resultado muestra que olvidar datos no siempre mejora
el desempeño y que la selección debe validarse temporalmente.

## Ranking local

| # | Método (notebook) | AUC local | Tiempo | Validación |
|---|---|---:|---:|---|
| 1 | XGBoost d12 + **magic UID** (`04`, técnica del 1.er puesto) | **0.9479** | 192 s GPU | GroupKFold mensual ×3 |
| 2 | XGBoost d12 sin UID (`04`) | 0.9375 | 163 s GPU | GroupKFold mensual ×3 |
| 3 | LightGBM + RFE + count encoding (`03`) | 0.9200 | 462 s CPU | TimeSeriesSplit ×3 |
| 4 | XGBoost GPU (`02`) | 0.9227 | 25 s GPU | holdout temporal 75/25 |
| 5 | XGBoost CPU baseline (`01`) | 0.9191 | 112 s CPU | holdout temporal 75/25 |
| + | **Sistema integrado** (`05`): d12 + magic UID + count encoding, semilla 42 | 0.9456 / 0.9509 | 320 s GPU | holdout 75/25 / OOF GroupKFold ×3 |

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
│   └── ejecutados/
│       ├── 01_xgboost_baseline_cpu.ipynb   baseline XGBoost CPU (ejecutado, con outputs)
│       ├── 02_xgboost_gpu.ipynb            XGBoost GPU (~40 s) (ejecutado)
│       ├── 03_lightgbm_encoding.ipynb      LightGBM + count encoding (ejecutado)
│       ├── 04_xgboost_magic_uid.ipynb      ablación magic UID (ejecutado)
│       ├── 05_sistema_integrado.ipynb      sistema propio integrado, semilla 42 (ejecutado)
│       └── _RESUMEN_TECNICAS.md            técnicas por notebook
├── src/                benchmarks y evaluación adaptativa
├── results/            CSV de resultados, ranking y técnicas (TECNICAS.md)
├── informe_latex/      informe largo, entrega <=8 páginas, figuras e infografía
├── presentacion/       presentación Beamer (.tex + PDF), figuras Python y versión .pptx
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
.venv/bin/python src/bench_01_inversion.py    # baseline CPU (= notebook 01)
.venv/bin/python src/bench_02_xhlulu.py       # baseline GPU (= notebook 02)
.venv/bin/python src/bench_03_nroman.py       # LightGBM (= notebook 03)
.venv/bin/python src/bench_04_cdeotte.py      # ablación magic UID (= notebook 04)

# 4. evaluación adaptativa por bloques temporales
.venv/bin/python src/adaptive_fraud.py \
  --windows expanding,30,60,90 --initial-days 60 --eval-days 30 \
  --model xgboost --device cuda --n-estimators 180

# 5. informe (figuras + PDF)
.venv/bin/python informe_latex/scripts/generar_graficos.py
.venv/bin/python informe_latex/scripts/generar_adaptacion.py
cd informe_latex && tectonic main.tex
# versión corta para entregar: main_entrega.pdf (3 páginas)
# infografía: figures/infografia_sistema_adaptativo.pdf
# 6. presentación Beamer (figuras 16:9 + PDF)
.venv/bin/python presentacion/generar_figuras.py
(cd presentacion && tectonic planifica_presentacion.tex)
# salida: presentacion/planifica_presentacion.pdf (14 páginas, 13 + apéndice)
```

Los gráficos del informe se regeneran desde `data/raw` y `results/`: si cambias los
benchmarks, re-ejecuta el paso 4.

## Limitación conocida

El benchmark original mantiene un protocolo transductivo para reproducir la competencia. La
evaluación adaptativa de `src/adaptive_fraud.py` implementa la variante **estrictamente causal**:
las estadísticas por UID solo usan transacciones anteriores a cada bloque futuro.

## Créditos y licencias

Código propio bajo [MIT](LICENSE). Los notebooks 01–04 son adaptaciones locales de
soluciones públicas de la competencia (inspiración, con modificaciones para ejecución
en local y actualización a las APIs actuales de XGBoost/LightGBM/pandas); conservan la
licencia Apache 2.0 de los originales. El notebook 05 (`05_sistema_integrado.ipynb`)
es creación propia del proyecto.

## Semillas

| Notebook / script | Semilla |
|---|---|
| `01_xgboost_baseline_cpu` | sin semilla explícita (XGBoost usa su default `seed=0`) |
| `02_xgboost_gpu` | `random_state=2019` |
| `03_lightgbm_encoding` | `random_state=47`, `bagging_seed=11` |
| `04_xgboost_magic_uid` | sin semilla explícita (split GroupKFold mensual, determinístico) |
| `05_sistema_integrado` | `SEED=42` |
| `src/adaptive_fraud.py` | `random_state=42` |
