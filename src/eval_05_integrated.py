"""Métricas de decisión (recall, precisión, F1) del sistema integrado 05.

El notebook 05 (`05_sistema_integrado.ipynb`) registra solo AUC. Este script
extiende ese experimento en dos direcciones y guarda cada una en su CSV:

A) `protocolo` — replica el notebook 05 tal cual (features calculadas sobre
   todo el train) y añade métricas de decisión en los umbrales 0.50 y 0.19,
   más el umbral que maximiza F1 en el OOF (marcado como in-sample del umbral).
   Debe reproducir los AUC de `results/bench_05_integrated.csv`.

B) `causal` — holdout 75/25 estrictamente causal: categorías, frecuencias y
   agregaciones UID se calculan SOLO con transacciones anteriores al corte.
   El umbral se calibra en el bloque [60%, 75%) y se aplica sobre el bloque
   futuro [75%, 100%]; nunca se elige el umbral donde se reporta.

Salidas:
    results/threshold_metrics_05_integrated.csv   (experimento A)
    results/threshold_metrics_05_causal.csv       (experimento B)

Ejemplos:
    .venv/bin/python src/eval_05_integrated.py --experiment protocolo
    .venv/bin/python src/eval_05_integrated.py --experiment causal
    .venv/bin/python src/eval_05_integrated.py --smoke   # prueba rápida CPU
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEED = 42
START_DATE = np.datetime64("2017-11-30T00:00:00")
FIXED_THRESHOLDS = (0.50, 0.19)

# ---------------------------------------------------------------------------
# definición de columnas (idéntica al notebook 05)
# ---------------------------------------------------------------------------

STR_TYPE = [
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29", "id_30",
    "id_31", "id_33", "id_34", "id_35", "id_36", "id_37", "id_38",
    "DeviceType", "DeviceInfo",
]
COLS = [
    "TransactionID", "TransactionDT", "TransactionAmt", "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2",
    "dist1", "dist2", "P_emaildomain", "R_emaildomain",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11",
    "C12", "C13", "C14",
    "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11",
    "D12", "D13", "D14", "D15",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
]
V = [1, 3, 4, 6, 8, 11] + [13, 14, 17, 20, 23, 26, 27, 30] + [36, 37, 40, 41, 44, 47, 48] + [54, 56, 59, 62, 65, 67, 68, 70]
V += [76, 78, 80, 82, 86, 88, 89, 91] + [107, 108, 111, 115, 117, 120, 121, 123] + [124, 127, 129, 130, 136]
V += [138, 139, 142, 147, 156, 162, 165, 160, 166] + [178, 176, 173, 182] + [187, 203, 205, 207, 215]
V += [169, 171, 175, 180, 185, 188, 198, 210, 209] + [218, 223, 224, 226, 228, 229, 235] + [240, 258, 257, 253, 252, 260, 261]
V += [264, 266, 267, 274, 277] + [220, 221, 234, 238, 250, 271] + [294, 284, 285, 286, 291, 297]
V += [303, 305, 307, 309, 310, 320] + [281, 283, 289, 296, 301, 314]
COLS += ["V" + str(x) for x in V]

DROP_FROM_MAGIC = [
    "D6", "D7", "D8", "D9", "D12", "D13", "D14", "C3", "M5",
    "id_08", "id_33", "card4", "id_07", "id_14", "id_21", "id_30", "id_32",
    "id_34",
] + ["id_" + str(x) for x in range(22, 28)]


def load_train(max_rows: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    dtypes = {c: "float32" for c in COLS + ["id_0" + str(i) for i in range(1, 10)] + ["id_" + str(i) for i in range(10, 34)]}
    for c in STR_TYPE:
        dtypes[c] = "category"
    X = pd.read_csv(
        ROOT / "data" / "raw" / "train_transaction.csv",
        index_col="TransactionID", dtype=dtypes, usecols=COLS + ["isFraud"],
    )
    tid = pd.read_csv(ROOT / "data" / "raw" / "train_identity.csv", index_col="TransactionID", dtype=dtypes)
    X = X.merge(tid, how="left", left_index=True, right_index=True)
    del tid
    if max_rows:
        X = X.iloc[:max_rows]
    y = X["isFraud"].copy()
    del X["isFraud"]
    gc.collect()
    print("Train:", X.shape, "| fraude:", f"{y.mean():.4%}")
    return X, y


# ---------------------------------------------------------------------------
# construcción de features (misma secuencia que el notebook 05)
# ---------------------------------------------------------------------------

def normalize_D(df: pd.DataFrame) -> None:
    for i in range(1, 16):
        if i in [1, 2, 3, 5, 9]:
            continue
        df["D" + str(i)] = df["D" + str(i)] - df.TransactionDT / np.float32(24 * 60 * 60)


def encode_basic_pair(past: pd.DataFrame, future: pd.DataFrame | None) -> None:
    """Codifica categóricas y desplaza numéricas.

    No causal (future is None): réplica exacta del notebook (factorize y min
    sobre todo el frame). Causal: categorías y mínimos aprendidos solo del
    pasado; lo no visto en el futuro mapea a -1.
    """
    frames = [past] if future is None else [past, future]
    for f in past.columns:
        if past[f].dtype.name in ("category", "object", "str"):
            codes, uniques = pd.factorize(past[f], sort=True)
            past[f] = codes.astype("int16")
            if future is not None:
                future[f] = pd.Categorical(future[f], categories=uniques).codes.astype("int16")
        elif f not in ("TransactionAmt", "TransactionDT"):
            mn = np.float32(past[f].min())
            if np.isnan(mn):
                mn = np.float32(0)
            for df in frames:
                df[f] = df[f] - mn
                df[f] = df[f].fillna(-1)


def _encode_FE(targets: list[pd.DataFrame], stat: pd.DataFrame, col: str) -> None:
    vc = stat[col].value_counts(dropna=True, normalize=True).to_dict()
    vc[-1] = -1
    for df in targets:
        df[col + "_FE"] = df[col].map(vc).astype("float32")
        df[col + "_FE"] = df[col + "_FE"].fillna(-1)


def _encode_CB(past: pd.DataFrame, future: pd.DataFrame | None, col1: str, col2: str) -> None:
    nm = col1 + "_" + col2
    frames = [past] if future is None else [past, future]
    for df in frames:
        df[nm] = df[col1].astype(str) + "_" + df[col2].astype(str)
    codes, uniques = pd.factorize(past[nm], sort=True)
    past[nm] = codes.astype("int32")
    if future is not None:
        mapping = {u: i for i, u in enumerate(uniques)}
        future[nm] = future[nm].map(mapping).astype("float32").fillna(-1).astype("int32")


def _encode_AG(targets: list[pd.DataFrame], stat: pd.DataFrame, main_columns, uids, aggregations, usena: bool) -> None:
    for main_column in main_columns:
        for col in uids:
            for agg_type in aggregations:
                nm = main_column + "_" + col + "_" + agg_type
                temp = stat[[col, main_column]].copy()
                if usena:
                    temp.loc[temp[main_column] == -1, main_column] = np.nan
                agg = temp.groupby(col)[main_column].agg([agg_type])
                del temp
                for df in targets:
                    df[nm] = df[col].map(agg[agg_type]).astype("float32")
                    df[nm] = df[nm].fillna(-1)


def _encode_AG2(targets: list[pd.DataFrame], stat: pd.DataFrame, main_columns, uids) -> None:
    for main_column in main_columns:
        for col in uids:
            mp = stat.groupby(col)[main_column].agg(["nunique"])["nunique"].to_dict()
            for df in targets:
                df[col + "_" + main_column + "_ct"] = df[col].map(mp).astype("float32")
                df[col + "_" + main_column + "_ct"] = df[col + "_" + main_column + "_ct"].fillna(-1)


def add_count_features(past: pd.DataFrame, future: pd.DataFrame | None) -> None:
    targets = [past] if future is None else [past, future]
    for df in targets:
        df["cents"] = (df["TransactionAmt"] - np.floor(df["TransactionAmt"])).astype("float32")
    _encode_FE(targets, past, "addr1")
    _encode_FE(targets, past, "card1")
    _encode_FE(targets, past, "card2")
    _encode_FE(targets, past, "card3")
    _encode_FE(targets, past, "P_emaildomain")
    _encode_CB(past, future, "card1", "card5")
    _encode_CB(past, future, "addr1", "card1")
    _encode_FE(targets, past, "card1_card5")
    _encode_FE(targets, past, "addr1_card1")


def add_uid_features(past: pd.DataFrame, future: pd.DataFrame | None) -> None:
    targets = [past] if future is None else [past, future]
    stat = past
    for df in targets:
        df["day"] = df.TransactionDT / (24 * 60 * 60)
        dt = pd.to_datetime(START_DATE) + pd.to_timedelta(df["TransactionDT"], unit="s")
        df["DT_M"] = (dt.dt.year - 2017) * 12 + dt.dt.month
        df["uid"] = df.addr1_card1.astype(str) + "_" + np.floor(df.day - df.D1).astype(str)
    _encode_FE(targets, stat, "uid")
    _encode_AG(targets, stat, ["TransactionAmt", "D4", "D9", "D10", "D15"], ["uid"], ["mean", "std"], usena=True)
    _encode_AG(targets, stat, ["C" + str(x) for x in range(1, 15) if x != 3], ["uid"], ["mean"], usena=True)
    _encode_AG(targets, stat, ["M" + str(x) for x in range(1, 10)], ["uid"], ["mean"], usena=True)
    _encode_AG2(targets, stat, ["P_emaildomain", "dist1", "DT_M", "id_02", "cents"], ["uid"])
    _encode_AG(targets, stat, ["C14"], ["uid"], ["std"], usena=True)
    _encode_AG2(targets, stat, ["C13", "V314"], ["uid"])
    _encode_AG2(targets, stat, ["V127", "V136", "V309", "V307", "V320"], ["uid"])
    for df in targets:
        df["outsider15"] = (np.abs(df.D1 - df.D15) > 3).astype("int8")


def feature_sets(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    cols_magic = [c for c in df.columns if c not in ("TransactionDT", "DT_M", "day", "uid")]
    for c in DROP_FROM_MAGIC:
        if c in cols_magic:
            cols_magic.remove(c)
    cols_base = [c for c in cols_magic if "uid" not in c and c != "outsider15"]
    return cols_base, cols_magic


# ---------------------------------------------------------------------------
# modelos y métricas
# ---------------------------------------------------------------------------

def xgb_params(args, n_estimators: int, esr: int) -> dict:
    return dict(
        n_estimators=n_estimators, max_depth=12, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.4, missing=-1,
        eval_metric="auc", early_stopping_rounds=esr,
        tree_method="hist", device=args.device, random_state=SEED,
    )


def decision_metrics(y_true: np.ndarray, proba: np.ndarray, thr: float) -> dict:
    pred = (proba >= thr).astype(np.int8)
    yt = np.asarray(y_true).astype(np.int8)
    tp = int(((pred == 1) & (yt == 1)).sum())
    fp = int(((pred == 1) & (yt == 0)).sum())
    fn = int(((pred == 0) & (yt == 1)).sum())
    tn = int(((pred == 0) & (yt == 0)).sum())
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return dict(
        threshold=round(float(thr), 4),
        accuracy=(tp + tn) / len(yt),
        recall=recall,
        precision=precision,
        f1=f1,
        balanced_accuracy=(recall + specificity) / 2,
    )


def f1_best_threshold(y_true, proba) -> tuple[float, float]:
    prec, rec, thr = precision_recall_curve(np.asarray(y_true), proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    i = int(np.nanargmax(f1[:-1]))
    return float(thr[i]), float(f1[i])


# ---------------------------------------------------------------------------
# experimento A: protocolo del notebook (comparable con bench_05_integrated)
# ---------------------------------------------------------------------------

def run_protocolo(args) -> None:
    print("=" * 70)
    print("EXPERIMENTO A — protocolo del notebook 05 (features sobre todo el train)")
    print("=" * 70)
    X, y = load_train(args.max_rows)
    normalize_D(X)
    encode_basic_pair(X, None)
    add_count_features(X, None)
    add_uid_features(X, None)
    cols_base, cols_magic = feature_sets(X)
    print("features sin UID:", len(cols_base), "| con magic UID:", len(cols_magic))
    assert len(cols_base) == 198 and len(cols_magic) == 245, "checksum de features no coincide con el notebook"

    rows: list[dict] = []
    registered = {}
    for feature_cols, tag in ((cols_base, "integrado 05 sin UID"), (cols_magic, "integrado 05 + magic UID")):
        t0 = time.time()
        cut = 3 * len(X) // 4
        idxT, idxV = X.index[:cut], X.index[cut:]
        clf = xgb.XGBClassifier(**xgb_params(args, args.n_estimators, 100))
        clf.fit(X.loc[idxT, feature_cols], y[idxT],
                eval_set=[(X.loc[idxV, feature_cols], y[idxV])], verbose=0)
        proba_h = clf.predict_proba(X.loc[idxV, feature_cols])[:, 1]
        auc_h = roc_auc_score(y[idxV], proba_h)
        del clf
        gc.collect()

        oof = np.zeros(len(X))
        n_groups = int(X["DT_M"].nunique())
        n_splits = 3 if not args.smoke else max(2, min(3, n_groups))
        gk = GroupKFold(n_splits=n_splits)
        for i, (tr, va) in enumerate(gk.split(X, y, groups=X["DT_M"])):
            clf = xgb.XGBClassifier(**{**xgb_params(args, 5000, 200), "n_estimators": args.n_estimators * 2})
            clf.fit(X[feature_cols].iloc[tr], y.iloc[tr],
                    eval_set=[(X[feature_cols].iloc[va], y.iloc[va])], verbose=0)
            oof[va] = clf.predict_proba(X[feature_cols].iloc[va])[:, 1]
            del clf
            gc.collect()
            print(f"[{tag}] fold {i + 1}/{n_splits} listo")
        auc_o = roc_auc_score(y, oof)
        seconds = time.time() - t0
        registered[tag] = (auc_h, auc_o)
        print(f"[{tag}] holdout={auc_h:.4f} | OOF={auc_o:.4f} | {seconds:.0f}s")

        for thr in FIXED_THRESHOLDS:
            rows.append({"model": tag, "validation": "holdout 75/25", "auc": round(auc_h, 4),
                         **decision_metrics(y[idxV].to_numpy(), proba_h, thr)})
            rows.append({"model": tag, "validation": f"OOF GroupKFold x{n_splits}", "auc": round(auc_o, 4),
                         **decision_metrics(y.to_numpy(), oof, thr)})
        tau, f1_star = f1_best_threshold(y, oof)
        row = {"model": tag, "validation": f"OOF GroupKFold x{n_splits}", "auc": round(auc_o, 4),
               **decision_metrics(y.to_numpy(), oof, tau)}
        row["notes"] = f"umbral calibrado en OOF (in-sample); F1 maximo {f1_star:.4f}"
        rows.append(row)
        gc.collect()

    if args.smoke:
        print("(smoke: se omite la verificacion contra bench_05_integrated.csv)")
    else:
        ref = pd.read_csv(RESULTS / "bench_05_integrated.csv")
        ok = (
            abs(ref.loc[0, "holdout_auc"] - registered["integrado 05 sin UID"][0]) < 5e-4
            and abs(ref.loc[0, "oof_auc"] - registered["integrado 05 sin UID"][1]) < 5e-4
            and abs(ref.loc[1, "holdout_auc"] - registered["integrado 05 + magic UID"][0]) < 5e-4
            and abs(ref.loc[1, "oof_auc"] - registered["integrado 05 + magic UID"][1]) < 5e-4
        )
        print("Verificacion vs bench_05_integrated.csv:", "OK" if ok else "DIFIERE (revisar)")

    out = pd.DataFrame(rows)[
        ["model", "validation", "threshold", "accuracy", "recall", "precision", "f1", "balanced_accuracy", "auc", "notes"]
    ]
    out["notes"] = out["notes"].fillna("umbrales fijos 0.50 y 0.19")
    out["source"] = "src/eval_05_integrated.py --experiment protocolo; features del notebook 05, semilla 42"
    RESULTS.mkdir(exist_ok=True)
    out.to_csv(RESULTS / "threshold_metrics_05_integrated.csv", index=False)
    print(out.to_string(index=False))
    print("guardado:", RESULTS / "threshold_metrics_05_integrated.csv")


# ---------------------------------------------------------------------------
# experimento B: causal con calibración de umbral
# ---------------------------------------------------------------------------

def run_causal(args) -> None:
    print("=" * 70)
    print("EXPERIMENTO B — holdout causal (agregados solo con el pasado)")
    print("=" * 70)
    X, y = load_train(args.max_rows)
    normalize_D(X)
    n = len(X)
    cut75 = 3 * n // 4
    cut60 = 3 * n // 5

    # fase 1: entrenar en [0,60%) y calibrar el umbral en [60%,75%)
    past1, cal = X.iloc[:cut60].copy(), X.iloc[cut60:cut75].copy()
    encode_basic_pair(past1, cal)
    add_count_features(past1, cal)
    add_uid_features(past1, cal)
    cols_base, cols_magic = feature_sets(past1)
    print("features sin UID:", len(cols_base), "| con magic UID:", len(cols_magic))

    clf = xgb.XGBClassifier(**xgb_params(args, args.n_estimators, 100))
    clf.fit(past1[cols_magic], y.iloc[:cut60],
            eval_set=[(cal[cols_magic], y.iloc[cut60:cut75])], verbose=0)
    proba_cal = clf.predict_proba(cal[cols_magic])[:, 1]
    auc_cal = roc_auc_score(y.iloc[cut60:cut75], proba_cal)
    tau, f1_cal = f1_best_threshold(y.iloc[cut60:cut75], proba_cal)
    print(f"calibracion [60%,75%): AUC={auc_cal:.4f} | umbral F1-optimo tau*={tau:.4f} (F1 {f1_cal:.4f})")
    del clf, past1, cal, proba_cal
    gc.collect()

    # fase 2: reentrenar en [0,75%) y evaluar en el futuro [75%,100%)
    past2, fut = X.iloc[:cut75].copy(), X.iloc[cut75:].copy()
    gc.collect()
    encode_basic_pair(past2, fut)
    add_count_features(past2, fut)
    add_uid_features(past2, fut)
    cols_base2, cols_magic2 = feature_sets(past2)
    assert cols_magic2 == cols_magic, "los conjuntos de features difieren entre fases"

    t0 = time.time()
    clf = xgb.XGBClassifier(**xgb_params(args, args.n_estimators, 100))
    clf.fit(past2[cols_magic], y.iloc[:cut75],
            eval_set=[(fut[cols_magic], y.iloc[cut75:])], verbose=0)
    proba_fut = clf.predict_proba(fut[cols_magic])[:, 1]
    auc_fut = roc_auc_score(y.iloc[cut75:], proba_fut)
    y_fut = y.iloc[cut75:].to_numpy()
    print(f"holdout causal: AUC={auc_fut:.4f} | {time.time() - t0:.0f}s")

    rows = []
    for thr in (*FIXED_THRESHOLDS, tau):
        row = {"model": "integrado 05 + magic UID (causal)",
               "validation": "holdout 75/25 causal",
               "auc": round(auc_fut, 4),
               **decision_metrics(y_fut, proba_fut, thr)}
        row["notes"] = (
            f"umbral calibrado en [60%,75%) sin usar el futuro (tau*={tau:.4f})"
            if thr == tau else "umbrales fijos 0.50 y 0.19"
        )
        rows.append(row)
    out = pd.DataFrame(rows)[
        ["model", "validation", "threshold", "accuracy", "recall", "precision", "f1", "balanced_accuracy", "auc", "notes"]
    ]
    out["source"] = "src/eval_05_integrated.py --experiment causal; features causales, semilla 42"
    RESULTS.mkdir(exist_ok=True)
    out.to_csv(RESULTS / "threshold_metrics_05_causal.csv", index=False)
    print(out.to_string(index=False))
    print("guardado:", RESULTS / "threshold_metrics_05_causal.csv")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", choices=["protocolo", "causal", "all"], default="all")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--max-rows", type=int, default=0, help="limita filas (depuracion)")
    ap.add_argument("--n-estimators", type=int, default=2000, help="arboles del holdout")
    ap.add_argument("--smoke", action="store_true", help="prueba rapida: pocas filas, CPU, sin OOF")
    args = ap.parse_args()
    if args.smoke:
        args.max_rows = args.max_rows or 200000
        args.device = "cpu"
        args.n_estimators = min(args.n_estimators, 120)
    np.random.seed(SEED)
    print("xgboost", xgb.__version__, "| device:", args.device, "| SEED =", SEED)
    if args.experiment in ("protocolo", "all"):
        run_protocolo(args)
    if args.experiment in ("causal", "all"):
        run_causal(args)


if __name__ == "__main__":
    main()
