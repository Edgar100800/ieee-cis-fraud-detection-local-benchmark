# Informe LaTeX — IEEE-CIS Fraud Detection

## Compilar

```bash
cd informe_latex
tectonic main.tex        # genera main.pdf (28 pag.)
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

- Figuras → `figures/*.pdf` (12)
- Tablas → `tables/tabla_datos.tex`, `tables/tabla_ranking.tex`

Si se re-ejecutan los benchmarks (`src/bench_0*.py`), correr de nuevo el
script de gráficos y recompilar para actualizar el informe.

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
