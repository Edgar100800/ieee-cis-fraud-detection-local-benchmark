"""Benchmark 2: xhlulu/ieee-fraud-xgboost-with-gpu-fit-in-40s.

Original: identico al baseline pero entrenando en GPU (gpu_hist).
Adaptacion XGBoost>=2.0: tree_method='hist' + device='cuda'.
"""
import time

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
import xgboost as xgb

print("XGBoost version:", xgb.__version__)
t0 = time.time()
train_transaction = pd.read_csv("data/raw/train_transaction.csv", index_col="TransactionID")
train_identity = pd.read_csv("data/raw/train_identity.csv", index_col="TransactionID")
train = train_transaction.merge(train_identity, how="left", left_index=True, right_index=True)
del train_transaction, train_identity
print("train:", train.shape)

y_train = train["isFraud"].copy()
X_train = train.drop("isFraud", axis=1)
del train
X_train = X_train.fillna(-999)

for f in X_train.columns:
    if not pd.api.types.is_numeric_dtype(X_train[f]):
        codes, _ = pd.factorize(X_train[f].astype(str), sort=True)
        X_train[f] = codes.astype(np.int32)

X_train = X_train.sort_index()
srt = X_train.index
y_train = y_train.loc[srt]
X_train = X_train.loc[srt]
cut = 3 * len(X_train) // 4
idxT, idxV = srt[:cut], srt[cut:]

clf = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=9,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    missing=-999,
    random_state=2019,
    eval_metric="auc",
    early_stopping_rounds=50,
    tree_method="hist",  # <- antes 'gpu_hist'
    device="cuda",  # <- THE MAGICAL PARAMETER (moderno)
)

t1 = time.time()
clf.fit(X_train.loc[idxT], y_train[idxT], eval_set=[(X_train.loc[idxV], y_train[idxV])], verbose=100)
fit_time = time.time() - t1

auc = roc_auc_score(y_train[idxV], clf.predict_proba(X_train.loc[idxV])[:, 1])
print(f"AUC validacion temporal: {auc:.6f}")
print(f"mejor iteracion: {clf.best_iteration}")
print(f"fit GPU: {fit_time:.1f}s | total: {time.time()-t0:.1f}s")

pd.DataFrame(
    [{"notebook": "xhlulu-xgb-gpu", "device": "gpu", "auc_holdout": auc,
      "fit_s": fit_time, "best_iter": int(clf.best_iteration), "n_features": X_train.shape[1]}]
).to_csv("results/bench_02_xhlulu.csv", index=False)
