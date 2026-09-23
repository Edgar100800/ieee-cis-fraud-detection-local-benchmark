"""Genera figuras 16:9 legibles para la presentación Beamer.

Fuentes:
- data/raw/train_transaction.csv
- results/RANKING_FINAL.csv
- results/bench_05_integrated.csv
- results/adaptive_results.csv
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "presentacion" / "figures"

INK = "#13263A"
TEAL = "#0F766E"
CORAL = "#D95B43"
SLATE = "#52606D"
MIST = "#EEF4F5"
BLUE = "#4C78A8"
LIGHT_BLUE = "#9BB7D4"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.edgecolor": SLATE,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": SLATE,
        "ytick.color": SLATE,
    }
)


def decimal(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight", dpi=180, facecolor="white")
    plt.close(fig)


def fraude_temporal() -> None:
    df = pd.read_csv(
        ROOT / "data" / "raw" / "train_transaction.csv",
        usecols=["TransactionDT", "isFraud"],
    )
    df["day"] = (df["TransactionDT"] // 86400).astype(int)
    daily = df.groupby("day")["isFraud"].mean().mul(100)
    global_rate = float(df["isFraud"].mean() * 100)

    fig, ax = plt.subplots(figsize=(12.8, 4.2))
    ax.plot(daily.index, daily.values, color=CORAL, linewidth=2.3)
    ax.axhline(global_rate, color=INK, linestyle="--", linewidth=1.7)
    ax.text(
        daily.index.max() - 2,
        global_rate + 0.22,
        f"media global {decimal(global_rate, 1)}%",
        ha="right",
        color=INK,
        fontsize=12,
        weight="bold",
    )
    ax.fill_between(daily.index, daily.values, global_rate, color=CORAL, alpha=0.08)
    ax.set_xlabel("día relativo desde el inicio del train")
    ax.set_ylabel("fraude diario (%)")
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "fraude_temporal")


def benchmark_auc() -> None:
    df = pd.read_csv(ROOT / "results" / "RANKING_FINAL.csv")
    values = {
        "XGB CPU baseline": float(df.loc[df["ranking"].eq(5), "auc"].iloc[0]),
        "LightGBM + FE": float(df.loc[df["ranking"].eq(3), "auc"].iloc[0]),
        "XGB GPU": float(df.loc[df["ranking"].eq(4), "auc"].iloc[0]),
        "XGB d12 sin UID": float(df.loc[df["ranking"].eq(2), "auc"].iloc[0]),
        "XGB d12 + magic UID": float(df.loc[df["ranking"].eq(1), "auc"].iloc[0]),
    }
    names = list(values)
    aucs = list(values.values())
    colors = [LIGHT_BLUE, LIGHT_BLUE, BLUE, INK, TEAL]

    fig, ax = plt.subplots(figsize=(11.8, 5.2))
    bars = ax.barh(names, aucs, color=colors, height=0.62)
    ax.set_xlim(0.915, 0.9525)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: decimal(x, 3)))
    ax.set_xlabel("AUC en validación temporal local")
    ax.grid(axis="x", alpha=0.2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for bar, value in zip(bars, aucs):
        ax.text(
            value + 0.00045,
            bar.get_y() + bar.get_height() / 2,
            decimal(value),
            va="center",
            fontsize=12,
            weight="bold" if value == max(aucs) else "normal",
            color=TEAL if value == max(aucs) else INK,
        )
    ax.annotate(
        "+0,0104",
        xy=(aucs[-1], 4),
        xytext=(aucs[-2], 4.42),
        arrowprops={"arrowstyle": "<->", "color": TEAL, "linewidth": 1.8},
        color=TEAL,
        fontsize=13,
        weight="bold",
        ha="center",
    )
    fig.tight_layout()
    save(fig, "benchmark_auc")


def ablacion_uid() -> None:
    df = pd.read_csv(ROOT / "results" / "bench_05_integrated.csv")
    holdout = df["holdout_auc"].to_numpy()
    oof = df["oof_auc"].to_numpy()
    x = np.arange(2)
    width = 0.32

    fig, ax = plt.subplots(figsize=(10.6, 5.2))
    bars_h = ax.bar(x - width / 2, holdout, width, color=SLATE, label="Holdout 75/25")
    bars_o = ax.bar(x + width / 2, oof, width, color=TEAL, label="OOF mensual ×3")
    ax.set_xticks(x, ["Sin UID", "+ magic UID"])
    ax.set_ylim(0.928, 0.954)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: decimal(y, 3)))
    ax.set_ylabel("AUC")
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.13))
    for bars in (bars_h, bars_o):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.0005,
                decimal(value),
                ha="center",
                fontsize=12,
                weight="bold",
            )
    ax.text(0.84, holdout[1] + 0.0016, f"+{decimal(holdout[1] - holdout[0])}", color=CORAL, weight="bold", fontsize=13, ha="center")
    ax.text(1.16, oof[1] + 0.0013, f"+{decimal(oof[1] - oof[0])}", color=TEAL, weight="bold", fontsize=13, ha="center")
    fig.tight_layout()
    save(fig, "ablacion_uid")


def _box(ax, x, y, w, h, text, face, fontsize=14, edge=INK, weight="bold") -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.5,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=INK, weight=weight)


def _arrow(ax, start, end, color=SLATE, connectionstyle=None) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=1.8,
            color=color,
            connectionstyle=connectionstyle,
        )
    )


def uid_diagrama() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 4.4))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    _box(ax, 0.2, 2.65, 2.3, 1.0, "card1 + addr1\ntarjeta + dirección", "#DCEEF2", 13)
    _box(ax, 0.2, 1.1, 2.3, 1.0, "floor(día − D1)\nfecha de alta", "#DCEEF2", 13)
    _box(ax, 3.35, 1.75, 2.1, 1.15, "UID\npseudo-cliente", "#F9F0C7", 15)
    _box(ax, 6.3, 3.0, 2.6, 0.78, "Frecuencia (FE)", "#DCEBD8", 13)
    _box(ax, 6.3, 1.85, 2.6, 0.78, "Media / std (AG)", "#DCEBD8", 13)
    _box(ax, 6.3, 0.7, 2.6, 0.78, "Cardinalidad (AG2)", "#DCEBD8", 13)
    _box(ax, 10.0, 1.6, 2.8, 1.25, "XGBoost d12\nsolo agregaciones", "#FBE4D5", 15)

    _arrow(ax, (2.5, 3.15), (3.35, 2.55))
    _arrow(ax, (2.5, 1.6), (3.35, 2.05))
    _arrow(ax, (5.45, 2.5), (6.3, 3.38))
    _arrow(ax, (5.45, 2.3), (6.3, 2.24))
    _arrow(ax, (5.45, 2.08), (6.3, 1.08))
    _arrow(ax, (8.9, 3.38), (10.0, 2.62))
    _arrow(ax, (8.9, 2.24), (10.0, 2.22))
    _arrow(ax, (8.9, 1.08), (10.0, 1.84))
    ax.text(4.4, 0.42, "El UID crudo se elimina antes del entrenamiento", ha="center", color=CORAL, fontsize=14, weight="bold")
    fig.tight_layout()
    save(fig, "uid_diagrama")


def pipeline_operativo() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 4.5))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    _box(ax, 0.3, 1.8, 3.35, 1.65, "1 · PREDICCIÓN\nDatos → features causales → XGBoost", "#DCEEF2", 14)
    _box(ax, 4.85, 1.8, 3.35, 1.65, "2 · DECISIÓN\nScore → umbral → acción", "#F9F0C7", 14)
    _box(ax, 9.4, 1.8, 3.35, 1.65, "3 · MONITOREO\nRecall + PSI + costo", "#DCEBD8", 14)
    _arrow(ax, (3.65, 2.62), (4.85, 2.62), TEAL)
    _arrow(ax, (8.2, 2.62), (9.4, 2.62), TEAL)
    _arrow(ax, (11.05, 1.8), (2.0, 1.18), CORAL, "arc3,rad=0.22")
    ax.text(6.55, 0.38, "Si cae el desempeño o sube el drift: revisar ventana y reentrenar", ha="center", fontsize=14, color=CORAL, weight="bold")
    fig.tight_layout()
    save(fig, "pipeline_operativo")


def protocolo_temporal() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 5.0))
    ax.set_xlim(0, 185)
    ax.set_ylim(-0.8, 5.2)
    ax.set_yticks([])
    ax.set_xlabel("día relativo")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.15)

    eval_start, eval_end = 150, 180
    rows = [
        ("Expansiva", 0, eval_start, TEAL),
        ("90 días", 60, eval_start, INK),
        ("60 días", 90, eval_start, BLUE),
        ("30 días", 120, eval_start, CORAL),
    ]
    for y, (label, start, end, color) in enumerate(rows[::-1], start=0):
        ax.add_patch(Rectangle((start, y), end - start, 0.62, facecolor=color, alpha=0.9))
        ax.text(start - 4, y + 0.31, label, ha="right", va="center", fontsize=13, weight="bold", color=INK)
    ax.add_patch(Rectangle((eval_start, 0), eval_end - eval_start, 3.62, facecolor="#F2D7DF", edgecolor=CORAL, linewidth=1.8))
    ax.text(165, 1.8, "Bloque futuro\n30 días", ha="center", va="center", fontsize=15, weight="bold", color=CORAL)
    ax.axvline(60, color=SLATE, linestyle="--", linewidth=1.4)
    ax.text(60, 4.35, "histórico inicial mínimo: 60 días", ha="center", fontsize=13, color=SLATE, weight="bold")
    ax.text(75, 3.92, "entrenamiento: solo pasado", ha="center", fontsize=13, color=TEAL, weight="bold")
    ax.text(150, 3.92, "corte temporal", ha="center", fontsize=13, color=CORAL, weight="bold")
    ax.set_xticks(np.arange(0, 181, 30))
    fig.tight_layout()
    save(fig, "protocolo_temporal")


def auc_adaptativo() -> None:
    df = pd.read_csv(ROOT / "results" / "adaptive_results.csv")
    colors = {"expanding": TEAL, "90d": INK, "60d": BLUE, "30d": CORAL}
    labels = {"expanding": "Expansiva", "90d": "90 días", "60d": "60 días", "30d": "30 días"}
    fig, ax = plt.subplots(figsize=(11.8, 5.2))
    for window in ["expanding", "90d", "60d", "30d"]:
        part = df[df["window"].eq(window)].sort_values("period")
        mean_auc = part["auc"].mean()
        ax.plot(
            part["period"],
            part["auc"],
            marker="o",
            markersize=8,
            linewidth=3 if window == "expanding" else 2.2,
            color=colors[window],
            label=f"{labels[window]} · media {decimal(mean_auc)}",
        )
    ax.axvspan(1.82, 2.18, color=CORAL, alpha=0.08)
    ax.annotate("caída común en bloque 2", xy=(2, 0.907), xytext=(2.35, 0.896), arrowprops={"arrowstyle": "->", "color": CORAL}, color=CORAL, fontsize=12, weight="bold")
    ax.set_xlim(0.85, 4.15)
    ax.set_ylim(0.89, 0.925)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("bloque futuro")
    ax.set_ylabel("AUC")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: decimal(y, 3)))
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="lower right", fontsize=11)
    fig.tight_layout()
    save(fig, "auc_adaptativo")


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    fraude_temporal()
    benchmark_auc()
    ablacion_uid()
    uid_diagrama()
    pipeline_operativo()
    protocolo_temporal()
    auc_adaptativo()
    print("Figuras generadas en", FIGURES)
