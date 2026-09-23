"""Evaluacion temporal adaptativa para IEEE-CIS Fraud Detection.

Este modulo convierte el benchmark estatico en un experimento de sistema
adaptativo. Cada periodo futuro se evalua con un modelo entrenado usando:

* una ventana expansiva, o
* una ventana deslizante de N dias;

y las agregaciones por UID se calculan usando solo transacciones anteriores.
El objetivo es medir degradacion, drift y costo de decisiones sin contaminar
el futuro con estadisticas calculadas sobre el bloque de evaluacion.

Ejemplo rapido:

    .venv/bin/python src/adaptive_fraud.py \
        --max-rows 120000 --device cpu --max-periods 2

Experimento completo (puede tardar varias horas):

    .venv/bin/python src/adaptive_fraud.py \
        --windows expanding,30,60,90 --device cuda
"""

from __future__ import annotations

import argparse
import gc
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
START_DATE = pd.Timestamp("2017-11-30")

# Este conjunto mantiene la idea del benchmark original, pero es ligero
# suficiente para repetir el experimento varias veces por periodo.
BASE_NUMERIC = [
    "TransactionAmt",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "C1",
    "C2",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "C9",
    "C10",
    "C11",
    "C12",
    "C13",
    "C14",
    "D1",
    "D2",
    "D4",
    "D5",
    "D9",
    "D10",
    "D11",
    "D15",
    "id_01",
    "id_02",
    "id_05",
    "id_06",
    "id_09",
    "id_11",
    "id_13",
    "id_14",
    "id_17",
    "id_19",
    "id_20",
    "id_32",
    "id_33",
]
BASE_CATEGORICAL = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
    "id_12",
    "id_15",
    "id_30",
    "id_31",
    "id_36",
    "id_37",
    "id_38",
    "DeviceType",
]

CAUSAL_AGG_COLUMNS = ["TransactionAmt", "D4", "D10", "D15", "C1", "C13"]
DRIFT_COLUMNS = ["TransactionAmt", "D1", "D4", "D10", "D15", "C1", "C13"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--windows",
        default="expanding,30,60,90",
        help="Ventanas en dias; usar expanding o una lista separada por comas.",
    )
    parser.add_argument("--initial-days", type=int, default=90)
    parser.add_argument("--eval-days", type=int, default=30)
    parser.add_argument("--max-periods", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument(
        "--model",
        choices=("xgboost", "logistic", "random_forest"),
        default="xgboost",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--false-positive-cost", type=float, default=1.0)
    parser.add_argument("--false-negative-cost", type=float, default=10.0)
    parser.add_argument("--output", default="results/adaptive_results.csv")
    return parser.parse_args()


def load_data(max_rows: int = 0) -> pd.DataFrame:
    tx_path = RAW / "train_transaction.csv"
    id_path = RAW / "train_identity.csv"
    if not tx_path.exists() or not id_path.exists():
        raise FileNotFoundError(
            "No se encontraron los datos. Ejecuta las instrucciones de data/README.md."
        )

    wanted = sorted(set(BASE_NUMERIC + BASE_CATEGORICAL + ["TransactionID", "TransactionDT", "isFraud"]))
    train = pd.read_csv(tx_path, usecols=lambda c: c in wanted)
    identity = pd.read_csv(id_path)
    train = train.merge(identity, on="TransactionID", how="left")
    train = train.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)
    if max_rows:
        train = train.iloc[:max_rows].copy()
    return train


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    day = df["TransactionDT"].astype("float64") / 86400.0
    timestamp = START_DATE + pd.to_timedelta(df["TransactionDT"], unit="s")
    df["transaction_day"] = day.astype("float32")
    df["transaction_hour"] = (df["TransactionDT"] // 3600 % 24).astype("int16")
    df["transaction_dow"] = timestamp.dt.dayofweek.astype("int8")
    df["calendar_month"] = (
        (timestamp.dt.year - START_DATE.year) * 12 + timestamp.dt.month
    ).astype("int16")

    for col in ["D4", "D5", "D10", "D11", "D15"]:
        if col in df:
            df[f"{col}_normalized"] = (
                pd.to_numeric(df[col], errors="coerce") - day
            ).astype("float32")

    card = df["card1"].astype("string").fillna("NA")
    addr = df["addr1"].astype("string").fillna("NA")
    d1 = pd.to_numeric(df["D1"], errors="coerce").fillna(-1)
    df["uid"] = card + "_" + addr + "_" + np.floor(day - d1).astype("int64").astype("string")
    return df


def add_causal_history_features(
    current: pd.DataFrame, history: pd.DataFrame | None
) -> pd.DataFrame:
    """Agrega estadisticas UID sin usar observaciones futuras.

    Para el entrenamiento se usa `shift` implícito mediante suma acumulada
    menos la fila actual. Para evaluación se construyen mapas exclusivamente
    con el histórico anterior al bloque evaluado.
    """

    df = current.copy()
    if history is None:
        group = df["uid"]
        valid = pd.Series(1, index=df.index, dtype="int64")
        count = valid.groupby(group, sort=False).cumsum() - 1
        df["uid_prior_count"] = count.astype("float32")
        for col in CAUSAL_AGG_COLUMNS:
            values = pd.to_numeric(df[col], errors="coerce")
            safe = values.fillna(0.0)
            cumulative = safe.groupby(group, sort=False).cumsum() - safe
            observed = values.notna().astype("int64")
            observed_prior = observed.groupby(group, sort=False).cumsum() - observed
            df[f"{col}_uid_prior_mean"] = (
                cumulative / observed_prior.replace(0, np.nan)
            ).astype("float32")
        return df

    hist = history[["uid"] + [c for c in CAUSAL_AGG_COLUMNS if c in history]].copy()
    grouped = hist.groupby("uid", sort=False)
    df["uid_prior_count"] = df["uid"].map(grouped.size()).fillna(0).astype("float32")
    for col in CAUSAL_AGG_COLUMNS:
        stats = grouped[col].agg(["mean"])["mean"]
        df[f"{col}_uid_prior_mean"] = df["uid"].map(stats).astype("float32")
    return df


def make_features(current: pd.DataFrame, history: pd.DataFrame | None) -> pd.DataFrame:
    df = add_time_features(current)
    df = add_causal_history_features(df, history)
    drop = {"TransactionID", "TransactionDT", "isFraud", "uid"}
    keep = [c for c in df.columns if c not in drop]
    return df[keep]


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = pd.to_numeric(expected, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    actual = pd.to_numeric(actual, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if expected.empty or actual.empty or expected.nunique() < 2:
        return 0.0
    edges = np.unique(np.nanquantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    e = pd.cut(expected, bins=edges, include_lowest=True).value_counts(sort=False, normalize=True)
    a = pd.cut(actual, bins=edges, include_lowest=True).value_counts(sort=False, normalize=True)
    e = np.clip(e.to_numpy(dtype=float), 1e-6, None)
    a = np.clip(a.to_numpy(dtype=float), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_summary(history: pd.DataFrame, current: pd.DataFrame) -> tuple[float, float, bool]:
    values = [psi(history[c], current[c]) for c in DRIFT_COLUMNS if c in history and c in current]
    if not values:
        return 0.0, 0.0, False
    mean_value = float(np.mean(values))
    max_value = float(np.max(values))
    return mean_value, max_value, bool(max_value >= 0.20)


def build_model(model_name: str, n_estimators: int, device: str):
    if model_name == "logistic":
        return Pipeline(
            [
                ("scale", StandardScaler(with_mean=False)),
                ("model", LogisticRegression(
                    max_iter=150,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                )),
            ]
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=14,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )

    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=9,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_weight=3,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        device=device,
        n_jobs=-1,
        random_state=42,
    )


def prepare_transformer(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]
    transformer = ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(strategy="median", add_indicator=True),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value", unknown_value=-1
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    return transformer, numeric, categorical


def metrics(y_true: pd.Series, proba: np.ndarray, threshold: float, fp_cost: float, fn_cost: float) -> dict[str, float]:
    pred = (proba >= threshold).astype("int8")
    tn = int(((y_true.to_numpy() == 0) & (pred == 0)).sum())
    fp = int(((y_true.to_numpy() == 0) & (pred == 1)).sum())
    fn = int(((y_true.to_numpy() == 1) & (pred == 0)).sum())
    tp = int(((y_true.to_numpy() == 1) & (pred == 1)).sum())
    cost = fp * fp_cost + fn * fn_cost
    return {
        "auc": float(roc_auc_score(y_true, proba)) if y_true.nunique() > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, proba)) if y_true.nunique() > 1 else float("nan"),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "false_positives": fp,
        "false_negatives": fn,
        "decision_cost": float(cost),
        "decision_utility": float(-cost),
        "true_positives": tp,
        "true_negatives": tn,
    }


def parse_windows(value: str) -> list[str | int]:
    windows: list[str | int] = []
    for token in value.split(","):
        token = token.strip().lower()
        if token == "expanding":
            windows.append(token)
        else:
            days = int(token)
            if days <= 0:
                raise ValueError("Las ventanas deben ser positivas.")
            windows.append(days)
    return windows


def run(args: argparse.Namespace) -> pd.DataFrame:
    data = load_data(args.max_rows)
    data = add_time_features(data)
    data["elapsed_day"] = data["transaction_day"] - data["transaction_day"].min()
    max_day = float(data["elapsed_day"].max())
    first_eval = float(args.initial_days)
    period = float(args.eval_days)
    if max_day <= first_eval:
        raise ValueError("No hay suficientes dias para evaluar despues del periodo inicial.")

    results: list[dict[str, float | str | int | bool]] = []
    windows = parse_windows(args.windows)
    period_number = 0
    eval_start = first_eval
    while eval_start < max_day and (not args.max_periods or period_number < args.max_periods):
        eval_end = min(eval_start + period, max_day + 1e-6)
        eval_mask = (data["elapsed_day"] >= eval_start) & (data["elapsed_day"] < eval_end)
        before_eval = data[data["elapsed_day"] < eval_start]
        current = data[eval_mask].copy()
        if current.empty:
            break
        period_number += 1
        print(f"Periodo {period_number}: dias {eval_start:.0f}-{eval_end:.0f}, filas={len(current)}")

        for window in windows:
            if window == "expanding":
                history = before_eval.copy()
                window_label = "expanding"
            else:
                start = eval_start - int(window)
                history = before_eval[before_eval["elapsed_day"] >= start].copy()
                window_label = f"{window}d"
            if history.empty or history["isFraud"].nunique() < 2 or current["isFraud"].nunique() < 2:
                print(f"  {window_label}: omitida por clases insuficientes")
                continue

            # Features de entrenamiento: estadisticas UID previas a cada fila.
            X_hist = make_features(history, history=None)
            X_current = make_features(current, history=history)
            y_hist = history["isFraud"].astype("int8")
            y_current = current["isFraud"].astype("int8")
            transformer, _, _ = prepare_transformer(X_hist)
            X_hist_enc = transformer.fit_transform(X_hist)
            X_current_enc = transformer.transform(X_current)
            model = build_model(args.model, args.n_estimators, args.device)
            started = time.time()
            model.fit(X_hist_enc, y_hist)
            proba = model.predict_proba(X_current_enc)[:, 1]
            fit_seconds = time.time() - started
            mean_psi, max_psi, drift_flag = drift_summary(history, current)
            row = {
                "period": period_number,
                "eval_start_day": eval_start,
                "eval_end_day": eval_end,
                "window": window_label,
                "model": args.model,
                "train_rows": len(history),
                "eval_rows": len(current),
                "fit_seconds": fit_seconds,
                "threshold": args.threshold,
                "mean_psi": mean_psi,
                "max_psi": max_psi,
                "drift_detected": drift_flag,
            }
            row.update(metrics(y_current, proba, args.threshold, args.false_positive_cost, args.false_negative_cost))
            results.append(row)
            print(
                f"  {window_label}: AUC={row['auc']:.4f} F1={row['f1']:.4f} "
                f"PSImax={max_psi:.3f} drift={drift_flag} ({fit_seconds:.1f}s)"
            )
            del transformer, model, X_hist_enc, X_current_enc, X_hist, X_current
            gc.collect()
        eval_start = eval_end

    result = pd.DataFrame(results)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Resultados adaptativos -> {output}")
    return result


if __name__ == "__main__":
    run(parse_args())
