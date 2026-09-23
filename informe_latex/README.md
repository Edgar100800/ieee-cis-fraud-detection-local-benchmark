# Informe LaTeX — IEEE-CIS Fraud Detection

## Compilar

```bash
cd informe_latex
tectonic main.tex         # genera main.pdf (informe técnico extenso)
tectonic main_entrega.tex # genera main_entrega.pdf (versión de entrega, <=8 pag.)
# alternativo: xelatex main.tex (x2 para referencias cruzadas)
```

Requisitos: `tectonic` (descarga paquetes automáticamente) o texlive completo.
`babel-spanish` es obligatorio; con texlive-básico instalar `texlive-langspanish`.

## Regenerar figuras y tablas

Los gráficos (PDF vectorial) y las tablas se generan desde los datos y
resultados reales del proyecto:

```bash
.venv/bin/python informe_latex/scripts/generar_graficos.py
```

- Figuras → `figures/*.pdf` (incluye arquitectura, métricas e infografía adaptativas)
- Tablas → `tables/tabla_datos.tex`, `tables/tabla_ranking.tex`

Si se re-ejecutan los benchmarks (`src/bench_0*.py`), correr de nuevo el
script de gráficos y recompilar para actualizar el informe.

La extensión adaptativa se ejecuta con:

```bash
cd ..
.venv/bin/python src/adaptive_fraud.py --windows expanding,30,60,90 \
  --initial-days 60 --eval-days 30 --model xgboost --device cuda
.venv/bin/python informe_latex/scripts/generar_adaptacion.py
.venv/bin/python informe_latex/scripts/generar_infografia.py
# métricas de decisión y variante causal del sistema integrado 05
.venv/bin/python src/eval_05_integrated.py --experiment all
```

Los resultados quedan en `results/adaptive_results.csv`,
`results/threshold_metrics_05_integrated.csv` y `results/threshold_metrics_05_causal.csv`;
las figuras en `figures/`.
La infografía final es `figures/infografia_sistema_adaptativo.pdf`.

## Estructura

```
main.tex            documento maestro (report, español, cleveref, listings)
referencias.bib     competencias, notebooks y papers (XGBoost, LightGBM)
chapters/
  introduccion.tex  motivación, objetivos, metodología
  datos.tex         dataset, desbalance, faltantes, estructura temporal
  tratamiento.tex   FE: tipos, faltantes, encoding, D-cols, UID, validación
  modelos.tex       configuraciones comparadas + costo computacional
  resultados.tex    ranking, ablación UID, importancias, referencias Kaggle
  discusion.tex     ventajas/límites, jerarquía de impacto, limitaciones
  conclusiones.tex  conclusiones + propuesta (UID causal, ensemble)
  apendice.tex      reproducibilidad, adaptaciones 2019→2026, ética
```
