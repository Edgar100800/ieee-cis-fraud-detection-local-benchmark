"""Benchmark 4: cdeotte/xgb-fraud-with-magic-0-9600 (tecnica del 1er puesto).

Original: XGB GPU max_depth=12, ~200 features (120 V cols seleccionadas + FE),
D-columns normalizadas, UID magico (card1+addr1+D1n) + agregaciones por grupo,
validacion GroupKFold por mes.
Adaptaciones: XGBoost>=2 (device='cuda'), split 75/25 temporal + GroupKFold 3 meses
(6 folds originales reducidos a 3 para RTX 2060 6GB).
"""
import datetime
import gc
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb

BUILD95 = True  # XGB sin magic (LB 0.95)
BUILD96 = True  # XGB con magic (LB 0.96)

t00 = time.time()

# COLUMNS WITH STRINGS
str_type = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 'M4', 'M5',
            'M6', 'M7', 'M8', 'M9', 'id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29', 'id_30',
            'id_31', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']

cols = ['TransactionID', 'TransactionDT', 'TransactionAmt',
        'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
        'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain',
        'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
        'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8',
        'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M1', 'M2', 'M3', 'M4',
        'M5', 'M6', 'M7', 'M8', 'M9']

v = [1, 3, 4, 6, 8, 11]
v += [13, 14, 17, 20, 23, 26, 27, 30]
v += [36, 37, 40, 41, 44, 47, 48]
v += [54, 56, 59, 62, 65, 67, 68, 70]
v += [76, 78, 80, 82, 86, 88, 89, 91]
v += [107, 108, 111, 115, 117, 120, 121, 123]
v += [124, 127, 129, 130, 136]
v += [138, 139, 142, 147, 156, 162]
v += [165, 160, 166]
v += [178, 176, 173, 182]
v += [187, 203, 205, 207, 215]
v += [169, 171, 175, 180, 185, 188, 198, 210, 209]
v += [218, 223, 224, 226, 228, 229, 235]
v += [240, 258, 257, 253, 252, 260, 261]
v += [264, 266, 267, 274, 277]
v += [220, 221, 234, 238, 250, 271]
v += [294, 284, 285, 286, 291, 297]
v += [303, 305, 307, 309, 310, 320]
v += [281, 283, 289, 296, 301, 314]

cols += ["V" + str(x) for x in v]
dtypes = {}
for c in cols + ["id_0" + str(x) for x in range(1, 10)] + ["id_" + str(x) for x in range(10, 34)]:
    dtypes[c] = "float32"
for c in str_type:
    dtypes[c] = "category"

# LOAD
t0 = time.time()
X_train = pd.read_csv("data/raw/train_transaction.csv", index_col="TransactionID", dtype=dtypes, usecols=cols + ["isFraud"])
train_id = pd.read_csv("data/raw/train_identity.csv", index_col="TransactionID", dtype=dtypes)
X_train = X_train.merge(train_id, how="left", left_index=True, right_index=True)
del train_id
y_train = X_train["isFraud"].copy()
del X_train["isFraud"]
gc.collect()
print(f"Train shape {X_train.shape} | load {time.time()-t0:.0f}s")

# NORMALIZE D COLUMNS
for i in range(1, 16):
    if i in [1, 2, 3, 5, 9]:
        continue
    X_train["D" + str(i)] = X_train["D" + str(i)] - X_train.TransactionDT / np.float32(24 * 60 * 60)

# FACTORIZE CATEGORICALS + SHIFT NUMERICS POSITIVE (NAN -> -1)
for i, f in enumerate(X_train.columns):
    if X_train[f].dtype == "category" or X_train[f].dtype == "object":
        df_comb = X_train[f]
        df_comb, _ = pd.factorize(df_comb, sort=True)
        X_train[f] = df_comb.astype("int16")
    elif f not in ["TransactionAmt", "TransactionDT"]:
        mn = X_train[f].min()
        X_train[f] -= np.float32(mn)
        X_train[f].fillna(-1, inplace=True)

# ENCODING FUNCTIONS (originales de Deotte, simplificadas a train-only para benchmark local)
def encode_FE(df1, cols):
    for col in cols:
        vc = df1[col].value_counts(dropna=True, normalize=True).to_dict()
        vc[-1] = -1
        nm = col + "_FE"
        df1[nm] = df1[col].map(vc).astype("float32")

def encode_AG(main_columns, uids, aggregations=["mean"], df=X_train, fillna=True, usena=False):
    for main_column in main_columns:
        for col in uids:
            for agg_type in aggregations:
                new_col_name = main_column + "_" + col + "_" + agg_type
                temp_df = df[[col, main_column]].copy()
                if usena:
                    temp_df.loc[temp_df[main_column] == -1, main_column] = np.nan
                temp_df = temp_df.groupby([col])[main_column].agg([agg_type]).reset_index()
                temp_df.columns = [col, new_col_name]
                mp = temp_df.set_index(col)[new_col_name].to_dict()
                df[new_col_name] = df[col].map(mp).astype("float32")
                if fillna:
                    df[new_col_name].fillna(-1, inplace=True)

def encode_CB(col1, col2, df=X_train):
    nm = col1 + "_" + col2
    df[nm] = df[col1].astype(str) + "_" + df[col2].astype(str)
    df_comb, _ = pd.factorize(df[nm], sort=True)
    df[nm] = df_comb.astype("int32")

def encode_AG2(main_columns, uids, df=X_train):
    for main_column in main_columns:
        for col in uids:
            comb = df[[col, main_column]]
            mp = comb.groupby(col)[main_column].agg(["nunique"])["nunique"].to_dict()
            df[col + "_" + main_column + "_ct"] = df[col].map(mp).astype("float32")

# FE base
X_train["cents"] = (X_train["TransactionAmt"] - np.floor(X_train["TransactionAmt"])).astype("float32")
encode_FE(X_train, ["addr1", "card1", "card2", "card3", "P_emaildomain"])
encode_CB("card1", "addr1")
encode_CB("card1_addr1", "P_emaildomain")
encode_FE(X_train, ["card1_addr1", "card1_addr1_P_emaildomain"])
encode_AG(["TransactionAmt", "D9", "D11"], ["card1", "card1_addr1", "card1_addr1_P_emaildomain"], ["mean", "std"], usena=True)

# Feature selection por time-consistency (original de Deotte)
cols95 = list(X_train.columns)
cols95.remove("TransactionDT")
for c in ["D6", "D7", "D8", "D9", "D12", "D13", "D14"]:
    cols95.remove(c)
for c in ["C3", "M5", "id_08", "id_33"]:
    cols95.remove(c)
for c in ["card4", "id_07", "id_14", "id_21", "id_30", "id_32", "id_34"]:
    cols95.remove(c)
for c in ["id_" + str(x) for x in range(22, 28)]:
    cols95.remove(c)

# Meses calendario para GroupKFold
START_DATE = datetime.datetime.strptime("2017-11-30", "%Y-%m-%d")
X_train["DT_M"] = X_train["TransactionDT"].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
X_train["DT_M"] = (X_train["DT_M"].dt.year - 2017) * 12 + X_train["DT_M"].dt.month


def fit_eval(feature_cols, tag):
    """75/25 temporal + GroupKFold 3 folds por mes."""
    # (a) holdout temporal 75/25
    idxT = X_train.index[: 3 * len(X_train) // 4]
    idxV = X_train.index[3 * len(X_train) // 4:]
    clf = xgb.XGBClassifier(
        n_estimators=2000, max_depth=12, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.4, missing=-1,
        eval_metric="auc", early_stopping_rounds=100,
        tree_method="hist", device="cuda",
    )
    t1 = time.time()
    clf.fit(X_train.loc[idxT, feature_cols], y_train[idxT],
            eval_set=[(X_train.loc[idxV, feature_cols], y_train[idxV])], verbose=200)
    auc_ho = roc_auc_score(y_train[idxV], clf.predict_proba(X_train.loc[idxV, feature_cols])[:, 1])
    ho_s = time.time() - t1
    print(f"[{tag}] holdout 75/25 AUC={auc_ho:.6f} best_iter={clf.best_iteration} ({ho_s:.0f}s)")
    del clf
    gc.collect()

    # (b) GroupKFold por mes (3 folds para GPU 6GB)
    oof = np.zeros(len(X_train))
    skf = GroupKFold(n_splits=3)
    t1 = time.time()
    for i, (tr, va) in enumerate(skf.split(X_train, y_train, groups=X_train["DT_M"])):
        clf = xgb.XGBClassifier(
            n_estimators=5000, max_depth=12, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.4, missing=-1,
            eval_metric="auc", early_stopping_rounds=200,
            tree_method="hist", device="cuda",
        )
        clf.fit(X_train[feature_cols].iloc[tr], y_train.iloc[tr],
                eval_set=[(X_train[feature_cols].iloc[va], y_train.iloc[va])], verbose=0)
        oof[va] += clf.predict_proba(X_train[feature_cols].iloc[va])[:, 1]
        del clf
        gc.collect()
        print(f"[{tag}] fold {i+1}/3 listo")
    auc_cv = roc_auc_score(y_train, oof)
    print(f"[{tag}] OOF GroupKFold(3 meses) AUC={auc_cv:.6f} ({time.time()-t1:.0f}s)")
    return auc_ho, auc_cv, ho_s, time.time() - t1


results = []
if BUILD95:
    auc_ho, auc_cv, s1, s2 = fit_eval(cols95, "XGB_95 (sin magic)")
    results.append({"notebook": "cdeotte-xgb95-sin-magic", "auc_holdout": auc_ho, "auc_oof_groupkfold3": auc_cv, "holdout_s": s1, "cv_s": s2})

# THE MAGIC: UID = card1+addr1 + floor(day - D1)
X_train["day"] = X_train.TransactionDT / (24 * 60 * 60)
X_train["uid"] = X_train.card1_addr1.astype(str) + "_" + np.floor(X_train.day - X_train.D1).astype(str)
encode_FE(X_train, ["uid"])
encode_AG(["TransactionAmt", "D4", "D9", "D10", "D15"], ["uid"], ["mean", "std"], fillna=True, usena=True)
encode_AG(["C" + str(x) for x in range(1, 15) if x != 3], ["uid"], ["mean"], fillna=True, usena=True)
encode_AG(["M" + str(x) for x in range(1, 10)], ["uid"], ["mean"], fillna=True, usena=True)
encode_AG2(["P_emaildomain", "dist1", "DT_M", "id_02", "cents"], ["uid"])
encode_AG(["C14"], ["uid"], ["std"], fillna=True, usena=True)
encode_AG2(["C13", "V314"], ["uid"])
encode_AG2(["V127", "V136", "V309", "V307", "V320"], ["uid"])
X_train["outsider15"] = (np.abs(X_train.D1 - X_train.D15) > 3).astype("int8")

cols96 = list(X_train.columns)
cols96.remove("TransactionDT")
for c in ["D6", "D7", "D8", "D9", "D12", "D13", "D14"]:
    cols96.remove(c)
for c in ["oof", "DT_M", "day", "uid"]:
    if c in cols96:
        cols96.remove(c)
for c in ["C3", "M5", "id_08", "id_33"]:
    cols96.remove(c)
for c in ["card4", "id_07", "id_14", "id_21", "id_30", "id_32", "id_34"]:
    cols96.remove(c)
for c in ["id_" + str(x) for x in range(22, 28)]:
    cols96.remove(c)

if BUILD96:
    auc_ho, auc_cv, s1, s2 = fit_eval(cols96, "XGB_96 (con magic UID)")
    results.append({"notebook": "cdeotte-xgb96-magic", "auc_holdout": auc_ho, "auc_oof_groupkfold3": auc_cv, "holdout_s": s1, "cv_s": s2})

print(f"\nTOTAL: {time.time()-t00:.0f}s")
pd.DataFrame(results).to_csv("results/bench_04_cdeotte.csv", index=False)
