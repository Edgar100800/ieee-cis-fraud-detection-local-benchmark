# Notebooks analizados sin ejecutar (descartados del benchmark)

Fuente: competencia [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/code).
Cada carpeta `codigos/descartados/` contiene el `.ipynb` original y su exportación `.py`.
Motivos de descarte: EDA pura (sin modelo entrenable) o infraestructura GPU no compatible
con 6 GB de VRAM (RAPIDS).

| Notebook | Votos | Motivo de descarte | Aporte analítico |
|---|---:|---|---|
| [artgor/eda-and-models](https://www.kaggle.com/code/artgor/eda-and-models) | 2986 | EDA + modelo no aislado; blend de 3 GBMs | Pipeline EDA→FE→blend por rangos |
| [kabure/extensive-eda-and-modeling-xgb-hyperopt](https://www.kaggle.com/code/kabure/extensive-eda-and-modeling-xgb-hyperopt) | 1907 | EDA con hyperopt, muy pesado | Normalización D-cols, frequency encoding, log del monto |
| [jesucristo/fraud-complete-eda](https://www.kaggle.com/code/jesucristo/fraud-complete-eda) | 858 | EDA pura | Análisis de NaN, bloques V, dominios de correo |
| [shahules/tackling-class-imbalance](https://www.kaggle.com/code/shahules/tackling-class-imbalance) | 858 | Catálogo de re-muestreo, no pipeline final | SMOTE/Tomek: útiles para aprender, no usados por los top |
| [robikscube/ieee-fraud-detection-first-look-and-eda](https://www.kaggle.com/code/robikscube/ieee-fraud-detection-first-look-and-eda) | 758 | EDA pura | Primer vistazo al dataset, split temporal |
| [vincentlugat/ieee-lgb-bayesian-opt](https://www.kaggle.com/code/vincentlugat/ieee-lgb-bayesian-opt) | 454 | Optimización bayesiana completa (horas de cómputo) | Flujo Bayes-opt para LGBM |
| [kyakovlev/ieee-gb-2-make-amount-useful-again](https://www.kaggle.com/code/kyakovlev/ieee-gb-2-make-amount-useful-again) | 579 | FE suelta, requiere el stack completo del autor | `TransactionAmt_check`, `log1p`, agregaciones de monto |
| [cdeotte/rapids-feature-engineering-fraud-0-96](https://www.kaggle.com/code/cdeotte/rapids-feature-engineering-fraud-0-96) | 169 | RAPIDS cuDF requiere >6 GB VRAM | Todo el FE en GPU (15× más rápido) |
