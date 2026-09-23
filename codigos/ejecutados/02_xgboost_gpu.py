#!/usr/bin/env python
# coding: utf-8

# # XGBoost con GPU — ajuste en ~40 s
# 
# Adaptación local de una solución pública de la competencia
# [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) (Kaggle, 2019).
# 
# Parte del baseline simple de XGBoost (etiquetado + `fillna(-999)`) y lo acelera:
# * Reduce el uso de RAM para no quedar sin memoria al usar GPU.
# * Reduce el **tiempo de ajuste de ~3500 s a ~40 s** usando GPU, con una ligera pérdida de AUC.
# 
# > Notebook inspirado en las soluciones públicas de la competencia (licencia Apache 2.0),
# > adaptado para ejecutarse en local con los datos de `data/raw/`.
# > Semilla: `random_state = 2019` (fijada en el clasificador para reproducibilidad).

# In[1]:


import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb


# In[2]:


print("XGBoost version:", xgb.__version__)


# # Efficient Preprocessing
# 
# This preprocessing method is more careful with RAM usage, which avoids crashing the kernel when you switch from CPU to GPU. Otherwise, it is exactly the same procedure as the official starter.

# In[3]:


get_ipython().run_cell_magic('time', '', "train_transaction = pd.read_csv('data/raw/train_transaction.csv', index_col='TransactionID')\ntest_transaction = pd.read_csv('data/raw/test_transaction.csv', index_col='TransactionID')\n\ntrain_identity = pd.read_csv('data/raw/train_identity.csv', index_col='TransactionID')\ntest_identity = pd.read_csv('data/raw/test_identity.csv', index_col='TransactionID')\n\nsample_submission = pd.read_csv('data/raw/sample_submission.csv', index_col='TransactionID')\n\ntrain = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)\ntest = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)\n# normaliza id-XX -> id_XX (quirk del CSV original de test_identity)\ntest.columns = [c.replace('id-', 'id_') for c in test.columns]\n\nprint(train.shape)\nprint(test.shape)\n\ny_train = train['isFraud'].copy()\ndel train_transaction, train_identity, test_transaction, test_identity\n\n# Drop target, fill in NaNs\nX_train = train.drop('isFraud', axis=1)\nX_test = test.copy()\n\ndel train, test\n\nX_train = X_train.fillna(-999)\nX_test = X_test.fillna(-999)\n\n# Label Encoding\nfor f in X_train.columns:\n    if not (pd.api.types.is_numeric_dtype(X_train[f]) and pd.api.types.is_numeric_dtype(X_test[f])):\n        lbl = preprocessing.LabelEncoder()\n        lbl.fit(list(X_train[f].values) + list(X_test[f].values))\n        X_train[f] = lbl.transform(list(X_train[f].values))\n        X_test[f] = lbl.transform(list(X_test[f].values))   \n")


# # Training
# 
# En XGBoost >= 2.0 la GPU se activa con `tree_method='hist'` + `device='cuda'`
# (el antiguo `tree_method='hist', device='cuda'` fue eliminado de la API).

# In[4]:


clf = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=9,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    missing=-999,
    random_state=2019,
    tree_method='hist', device='cuda'  # THE MAGICAL PARAMETER
)


# In[5]:


get_ipython().run_line_magic('time', 'clf.fit(X_train, y_train)')


# La ganancia de tiempo no viene solo de la GPU: también se calcula una **aproximación**
# del algoritmo exacto (que es greedy), lo que reduce levemente el AUC a cambio de mucha velocidad.
# 
# En CPU con `tree_method='hist'` el ajuste tarda varios minutos, lejos del tiempo de GPU.
# Los [parámetros de XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html) documentan
# `tree_method` y `device` en detalle.

# In[6]:


sample_submission['isFraud'] = clf.predict_proba(X_test)[:,1]
sample_submission.to_csv('simple_xgboost.csv')

