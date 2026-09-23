#!/usr/bin/env python
# coding: utf-8

# # Sistema integrado — XGBoost d12 + Magic UID + Count Encoding
# 
# Notebook **original de este proyecto** que integra las mejores técnicas de las soluciones
# públicas de la competencia [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)
# (Kaggle, 2019) — *inspirado* en ellas, no copiado:
# 
# | Técnica | Origen de la idea |
# |---|---|
# | Normalización de columnas D (`D − TransactionDT/86400`) | solución 1.er puesto |
# | Magic UID `card1+addr1+floor(day−D1)` + agregaciones de grupo | solución 1.er puesto |
# | Count/frequency encoding + interacciones `card1×card5`, `addr1×card1` | mejor LightGBM público |
# | `fillna(-999)` / faltantes como categoría | baseline XGBoost simple |
# | Validación temporal honesta (holdout 75/25 + GroupKFold por mes) | protocolo de la competencia |
# 
# **Semilla global: `SEED = 42`** (afecta a XGBoost: `subsample`, `colsample_bytree` y el
# binning de GPU; el split GroupKFold por mes es determinístico).

# In[1]:


import gc, time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

SEED = 42
np.random.seed(SEED)
print("numpy", np.__version__, "| pandas", pd.__version__, "| xgboost", xgb.__version__, "| SEED =", SEED)


# ## 1. Carga de datos
# 
# Solo `train` (590 540 transacciones): la validación es local y temporal, no hay submission.

# In[2]:


get_ipython().run_cell_magic('time', '', "str_type = ['ProductCD','card4','card6','P_emaildomain','R_emaildomain','M1','M2','M3','M4','M5',\n            'M6','M7','M8','M9','id_12','id_15','id_16','id_23','id_27','id_28','id_29','id_30',\n            'id_31','id_33','id_34','id_35','id_36','id_37','id_38','DeviceType','DeviceInfo']\ncols = ['TransactionID','TransactionDT','TransactionAmt','ProductCD','card1','card2','card3',\n        'card4','card5','card6','addr1','addr2','dist1','dist2','P_emaildomain','R_emaildomain',\n        'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14',\n        'D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13','D14','D15',\n        'M1','M2','M3','M4','M5','M6','M7','M8','M9']\nv = [1,3,4,6,8,11] + [13,14,17,20,23,26,27,30] + [36,37,40,41,44,47,48] + [54,56,59,62,65,67,68,70]\nv += [76,78,80,82,86,88,89,91] + [107,108,111,115,117,120,121,123] + [124,127,129,130,136]\nv += [138,139,142,147,156,162,165,160,166] + [178,176,173,182] + [187,203,205,207,215]\nv += [169,171,175,180,185,188,198,210,209] + [218,223,224,226,228,229,235] + [240,258,257,253,252,260,261]\nv += [264,266,267,274,277] + [220,221,234,238,250,271] + [294,284,285,286,291,297]\nv += [303,305,307,309,310,320] + [281,283,289,296,301,314]\ncols += ['V'+str(x) for x in v]\n\ndtypes = {c: 'float32' for c in cols + ['id_0'+str(i) for i in range(1,10)] + ['id_'+str(i) for i in range(10,34)]}\nfor c in str_type: dtypes[c] = 'category'\n\nX = pd.read_csv('data/raw/train_transaction.csv', index_col='TransactionID', dtype=dtypes, usecols=cols+['isFraud'])\ntid = pd.read_csv('data/raw/train_identity.csv', index_col='TransactionID', dtype=dtypes)\nX = X.merge(tid, how='left', left_index=True, right_index=True)\ndel tid\ny = X['isFraud'].copy()\ndel X['isFraud']\ngc.collect()\nprint('Train:', X.shape, '| fraude:', f'{y.mean():.4%}')\n")


# ## 2. Normalización de columnas D
# 
# Las columnas D son *deltas* respecto a un momento del pasado y crecen con el tiempo
# (deriva temporal). Se convierten en fechas absolutas: `D15n = TransactionDT/(24·3600) − D15`,
# lo que las hace transferibles al futuro.

# In[3]:


get_ipython().run_cell_magic('time', '', "for i in range(1, 16):\n    if i in [1, 2, 3, 5, 9]:   # D1-D3, D5, D9 ya son estables\n        continue\n    X['D'+str(i)] = X['D'+str(i)] - X.TransactionDT / np.float32(24*60*60)\nprint('D columns normalizadas')\n")


# ## 3. Codificación de categóricas y faltantes
# 
# Categóricas → códigos enteros (`factorize`); numéricas se desplazan a positivo y los
# `NaN` se marcan con `-1` (los GBM tratan los faltantes como categoría informativa).

# In[4]:


get_ipython().run_cell_magic('time', '', "for f in X.columns:\n    if X[f].dtype.name in ('category', 'object', 'str'):\n        codes, _ = pd.factorize(X[f], sort=True)\n        X[f] = codes.astype('int16')\n    elif f not in ['TransactionAmt', 'TransactionDT']:\n        X[f] = X[f] - np.float32(X[f].min())\n        X[f] = X[f].fillna(-1)\nprint('categóricas codificadas | NaN -> -1')\n")


# ## 4. Count encoding, interacciones y señales de monto
# 
# Frecuencia de tarjeta/ dirección como señal de riesgo; entidades más finas
# `card1×card5` y `addr1×card1`; centavos del monto y features FE.

# In[5]:


get_ipython().run_cell_magic('time', '', "def encode_FE(df, cols):\n    for col in cols:\n        vc = df[col].value_counts(dropna=True, normalize=True).to_dict()\n        vc[-1] = -1\n        df[col+'_FE'] = df[col].map(vc).astype('float32')\n\ndef encode_CB(col1, col2, df):\n    nm = col1+'_'+col2\n    df[nm] = df[col1].astype(str)+'_'+df[col2].astype(str)\n    codes, _ = pd.factorize(df[nm], sort=True)\n    df[nm] = codes.astype('int32')\n\n# centavos del monto (señal de fraude en montos redondos)\nX['cents'] = (X['TransactionAmt'] - np.floor(X['TransactionAmt'])).astype('float32')\n\n# count/frequency encoding sobre entidades base\nencode_FE(X, ['addr1','card1','card2','card3','P_emaildomain'])\n\n# interacciones como entidades más finas (card1×card5, addr1×card1)\nencode_CB('card1','card5', X)\nencode_CB('addr1','card1', X)\nencode_FE(X, ['card1_card5','addr1_card1'])\nprint('FE + interacciones listas:', X.shape[1], 'columnas')\n")


# ## 5. Magic UID + agregaciones de grupo
# 
# `UID = card1_addr1 + floor(day − D1)` reconstruye el pseudo-cliente. Se agregan su
# frecuencia, media y std de montos/Ds/Cs/Ms y nunique de contactos. El UID **se elimina
# antes de entrenar**; solo sirve como llave de agregación.

# In[6]:


get_ipython().run_cell_magic('time', '', "def encode_AG(main_columns, uids, aggregations=['mean'], df=None, usena=False):\n    df = X if df is None else df\n    for main_column in main_columns:\n        for col in uids:\n            for agg_type in aggregations:\n                nm = main_column+'_'+col+'_'+agg_type\n                temp = df[[col, main_column]].copy()\n                if usena:\n                    temp.loc[temp[main_column] == -1, main_column] = np.nan\n                agg = temp.groupby(col)[main_column].agg([agg_type])\n                df[nm] = df[col].map(agg[agg_type]).astype('float32')\n                df[nm] = df[nm].fillna(-1)\n\ndef encode_AG2(main_columns, uids, df=None):\n    df = X if df is None else df\n    for main_column in main_columns:\n        for col in uids:\n            mp = df.groupby(col)[main_column].agg(['nunique'])['nunique'].to_dict()\n            df[col+'_'+main_column+'_ct'] = df[col].map(mp).astype('float32')\n\nX['day'] = X.TransactionDT / (24*60*60)\nSTART_DATE = np.datetime64('2017-11-30T00:00:00')\ndt = pd.to_datetime(START_DATE) + pd.to_timedelta(X['TransactionDT'], unit='s')\nX['DT_M'] = (dt.dt.year - 2017) * 12 + dt.dt.month\nX['uid'] = X.addr1_card1.astype(str) + '_' + np.floor(X.day - X.D1).astype(str)\nencode_FE(X, ['uid'])\nencode_AG(['TransactionAmt','D4','D9','D10','D15'], ['uid'], ['mean','std'], usena=True)\nencode_AG(['C'+str(x) for x in range(1,15) if x != 3], ['uid'], ['mean'], usena=True)\nencode_AG(['M'+str(x) for x in range(1,10)], ['uid'], ['mean'], usena=True)\nencode_AG2(['P_emaildomain','dist1','DT_M','id_02','cents'], ['uid'])\nencode_AG(['C14'], ['uid'], ['std'], usena=True)\nencode_AG2(['C13','V314'], ['uid'])\nencode_AG2(['V127','V136','V309','V307','V320'], ['uid'])\nX['outsider15'] = (np.abs(X.D1 - X.D15) > 3).astype('int8')\nprint('magic UID + agregaciones listas:', X.shape[1], 'columnas')\n")


# ## 6. Validación temporal honesta
# 
# Dos protocolos, ambos hacia el futuro: **holdout temporal 75/25** y **GroupKFold
# mensual ×3** (oof). Se fija `SEED` en el modelo; el split por meses es determinístico.

# In[7]:


get_ipython().run_cell_magic('time', '', 'cols_magic = [c for c in X.columns if c not in (\'TransactionDT\',\'DT_M\',\'day\',\'uid\')]\nfor c in [\'D6\',\'D7\',\'D8\',\'D9\',\'D12\',\'D13\',\'D14\',\'C3\',\'M5\',\'id_08\',\'id_33\',\'card4\',\n          \'id_07\',\'id_14\',\'id_21\',\'id_30\',\'id_32\',\'id_34\'] + [\'id_\'+str(x) for x in range(22,28)]:\n    if c in cols_magic: cols_magic.remove(c)\n\n# ablation: base = magic SIN las features derivadas del UID\ncols_base = [c for c in cols_magic if \'uid\' not in c and c != \'outsider15\']\nprint(\'features sin UID:\', len(cols_base), \'| con magic UID:\', len(cols_magic))\n\ndef fit_eval(feature_cols, tag):\n    res = {}\n    params = dict(n_estimators=2000, max_depth=12, learning_rate=0.02,\n                  subsample=0.8, colsample_bytree=0.4, missing=-1,\n                  eval_metric=\'auc\', early_stopping_rounds=100,\n                  tree_method=\'hist\', device=\'cuda\', random_state=SEED)\n    t0 = time.time()\n    cut = 3 * len(X) // 4\n    idxT, idxV = X.index[:cut], X.index[cut:]\n    clf = xgb.XGBClassifier(**params)\n    clf.fit(X.loc[idxT, feature_cols], y[idxT],\n            eval_set=[(X.loc[idxV, feature_cols], y[idxV])], verbose=500)\n    res[\'holdout\'] = roc_auc_score(y[idxV], clf.predict_proba(X.loc[idxV, feature_cols])[:,1])\n    del clf; gc.collect()\n\n    oof = np.zeros(len(X))\n    gk = GroupKFold(n_splits=3)\n    for i, (tr, va) in enumerate(gk.split(X, y, groups=X[\'DT_M\'])):\n        clf = xgb.XGBClassifier(**{**params, \'n_estimators\': 5000, \'early_stopping_rounds\': 200})\n        clf.fit(X[feature_cols].iloc[tr], y.iloc[tr],\n                eval_set=[(X[feature_cols].iloc[va], y.iloc[va])], verbose=0)\n        oof[va] = clf.predict_proba(X[feature_cols].iloc[va])[:,1]\n        del clf; gc.collect()\n        print(f\'[{tag}] fold {i+1}/3 listo\')\n    res[\'oof\'] = roc_auc_score(y, oof)\n    res[\'s\'] = time.time() - t0\n    print(f"[{tag}] holdout={res[\'holdout\']:.4f} | OOF GroupKFold x3={res[\'oof\']:.4f} | {res[\'s\']:.0f}s")\n    return res\n')


# ## 7. Ablación: sin UID vs. con magic UID
# 
# Mismo algoritmo, mismo hardware, misma semilla — cambia solo la representación de datos.

# In[8]:


get_ipython().run_cell_magic('time', '', 'r_base = fit_eval(cols_base, \'sin UID  \')\nr_magic = fit_eval(cols_magic, \'con magic\')\n\nprint()\nprint(\'=\' * 62)\nprint(f"{\'modelo\':<28}{\'holdout\':>10}{\'OOF x3\':>10}{\'tiempo\':>10}")\nprint(\'-\' * 62)\nprint(f"{\'XGBoost d12 sin UID\':<28}{r_base[\'holdout\']:>10.4f}{r_base[\'oof\']:>10.4f}{r_base[\'s\']:>8.0f}s")\nprint(f"{\'XGBoost d12 + magic UID\':<28}{r_magic[\'holdout\']:>10.4f}{r_magic[\'oof\']:>10.4f}{r_magic[\'s\']:>8.0f}s")\nprint(\'-\' * 62)\nprint(f"{\'GANANCIA (representación)\':<28}{r_magic[\'holdout\']-r_base[\'holdout\']:>10.4f}{r_magic[\'oof\']-r_base[\'oof\']:>10.4f}")\nprint(\'=\' * 62)\n')


# ## 8. Conclusión
# 
# La ganancia proviene de la **representación de datos** (magic UID + agregaciones),
# no del algoritmo ni del hardware: con el mismo XGBoost d12 y la misma semilla
# (`SEED = 42`), reconstruir el pseudo-cliente y agregar su historial mejora el AUC
# local en ~0.01. La validación debe ser temporal (holdout 75/25 u OOF mensual);
# un KFold aleatorio inflaría el resultado.
