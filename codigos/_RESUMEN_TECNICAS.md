# Resumen de tecnicas por notebook


## artgor__eda-and-models
- Celdas de codigo: 38
- Seciones: General information, Functions used in this kernel, Data loading and overview, Data Exploration, Feature engineering, Prepare data for modelling, LGBM, Blending
- Tecnicas: LightGBM(6), XGBoost(13), CatBoost(14), GPU (device=cuda/gpu)(1), Validacion temporal(2), KFold / StratifiedKFold(3), Feature agregacion (groupby)(32), UID / magic client(3), Label encoding(2), Reduccion de memoria(6), Ensemble/blend(1), EDA con matplotlib/seaborn(21)

## cdeotte__rapids-feature-engineering-fraud-0-96
- Celdas de codigo: 17
- Seciones: RAPIDS - Feature Engineering - 1st Place Fraud Comp - [0.96], Install RAPIDS, GPU Load Data, GPU Preprocess, GPU Encoding Functions, GPU Feature Engineering, Local Holdout Validation, Cross Validation and Inference, Submit to Kaggle
- Tecnicas: XGBoost(14), GPU (device=cuda/gpu)(2), KFold / StratifiedKFold(2), UID / magic client(30), Reduccion de memoria(12), EDA con matplotlib/seaborn(9)

## cdeotte__xgb-fraud-with-magic-0-9600
- Celdas de codigo: 28
- Seciones: XGB Fraud with Magic scores LB 0.96, How the Magic Works, Load Data, Normalize D Columns, Encoding Functions, Feature Engineering, Feature Selection - Time Consistency, Local Validation, Predict test.csv, Kaggle Submission File XGB_95, The Magic Feature - UID, Group Aggregation Features, Local Validation, Predict test.csv, Kaggle Submission File XGB_96
- Tecnicas: XGBoost(11), GPU (device=cuda/gpu)(4), KFold / StratifiedKFold(3), Feature agregacion (groupby)(2), UID / magic client(48), Reduccion de memoria(13), EDA con matplotlib/seaborn(40)

## inversion__ieee-simple-xgboost
- Celdas de codigo: 8
- Seciones: -
- Tecnicas: XGBoost(4), Label encoding(1)

## jesucristo__fraud-complete-eda
- Celdas de codigo: 84
- Seciones: Content, References:, Data, 1st problem: NaN, 2nd Problem ..., Time vs fe, isFraud vs time, C features: C1, C2 ... C14, D features: D1 ... D15, M features: M1 .. M9, V150, Groups, TransactionAmt, Unique Values, D Features
- Tecnicas: LightGBM(1), XGBoost(8), CatBoost(1), GPU (device=cuda/gpu)(2), KFold / StratifiedKFold(2), Label encoding(1), Frequency encoding(11), Reduccion de memoria(10), EDA con plotly(13), EDA con matplotlib/seaborn(105)

## kabure__extensive-eda-and-modeling-xgb-hyperopt
- Celdas de codigo: 69
- Seciones: As my other kernel has running very slow because the interactive plots, I decided to start again using only Seaborn and matplotlib., Competition Objective is to detect fraud in transactions; , Data, Questions, I hope you enjoy my kernel and if it be useful for you, <b>upvote</b> the kernel, Importing necessary libraries, Importing train datasets, Knowing the data, Target Distribution, Transaction Amount Quantiles, Ploting Transaction Amount Values Distribution, Seeing the Quantiles of Fraud and No Fraud Transactions, Transaction Amount Outliers, Now, let's known the Product Feature, Card Features
- Tecnicas: XGBoost(8), GPU (device=cuda/gpu)(2), Validacion temporal(2), KFold / StratifiedKFold(2), Hyperopt/Optuna/Bayes(5), Feature agregacion (groupby)(8), Normalizacion D-cols(4), Label encoding(1), Frequency encoding(19), Log transform amount(3), Reduccion de memoria(10), EDA con plotly(16), EDA con matplotlib/seaborn(126)

## kyakovlev__ieee-gb-2-make-amount-useful-again
- Celdas de codigo: 22
- Seciones: -
- Tecnicas: LightGBM(5), KFold / StratifiedKFold(1), Feature agregacion (groupby)(2), UID / magic client(22), Label encoding(2), Frequency encoding(3), Log transform amount(2)

## nroman__lgb-single-model-lb-0-9419
- Celdas de codigo: 19
- Seciones: -
- Tecnicas: LightGBM(6), Validacion temporal(2), UID / magic client(2), Label encoding(3), EDA con matplotlib/seaborn(7)

## robikscube__ieee-fraud-detection-first-look-and-eda
- Celdas de codigo: 37
- Seciones: IEEE Fraud Detection Competition, Data, Train vs Test are Time Series Split, Distribution of Target in Training Set, TransactionAmt, ProductCD, Categorical Features - Transaction, card1 - card6, addr1 & addr2, dist1 & dist2, C1 - C14, D1-D9, M1-M9, V1 - V339, Identity Data
- Tecnicas: Log transform amount(2), EDA con matplotlib/seaborn(34)

## shahules__tackling-class-imbalance
- Celdas de codigo: 37
- Seciones: [Loading Required libraries](#1)<a id="1"></a> <br>, [Loading Data](#2)<a id="2"></a> <br>, [The metric trap](#3)<a id="3"></a> <br>, [Merging transaction and identity dataset](#4)<a id="4"></a> <br>, [Resampling](#5)<a id="5"></a> <br>, [Resampling Techniques using sklearn](#6)<a id="6"></a> <br>, [Dimensionality Reduction and Clustering](#7)<a id="7"></a> <br>, [Python imbalanced-learn module](#8)<a id='8'></a></br>, [Under-sampling: Tomek links](#9), [Algorithmic Ensemble Techniques](#9)<a id="1"></a> <br>, WORK IN PROGRESS
- Tecnicas: XGBoost(2), SMOTE / oversampling(12), Label encoding(1), Frequency encoding(6), Reduccion de memoria(5), EDA con matplotlib/seaborn(28)

## vincentlugat__ieee-lgb-bayesian-opt
- Celdas de codigo: 31
- Seciones: <a id='1'>1. Librairies and data</a> , DATASETS, MERGE, MISSING VALUE, FILL NA, ENCODING, <a id='2'>2. Bayesian Optimisation</a> , CONFUSION MATRIX, <a id='3'>3. LGB + best hyperparameters</a> , <a id='4'>4. Features importance</a> , <a id='5'>5. Submission</a> 
- Tecnicas: LightGBM(8), SMOTE / oversampling(2), KFold / StratifiedKFold(2), Hyperopt/Optuna/Bayes(3), Label encoding(2), EDA con matplotlib/seaborn(41)

## xhlulu__ieee-fraud-xgboost-with-gpu-fit-in-40s
- Celdas de codigo: 6
- Seciones: About this kernel, Changelog, Efficient Preprocessing, Training
- Tecnicas: XGBoost(6), GPU (device=cuda/gpu)(1), Label encoding(1)