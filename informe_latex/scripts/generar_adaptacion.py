"""Genera las figuras del sistema adaptativo a partir de results/adaptive_results.csv."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "informe_latex" / "figures"
RESULTS = ROOT / "results"


def metricas() -> None:
    df = pd.read_csv(RESULTS / "adaptive_results.csv")
    colors = {
        "expanding": "#173f5f",
        "30d": "#ed553b",
        "60d": "#3caea3",
        "90d": "#20639b",
    }
    labels = {
        "expanding": "Expansiva",
        "30d": "Ventana 30 días",
        "60d": "Ventana 60 días",
        "90d": "Ventana 90 días",
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for ax, metric, title in zip(
        axes,
        ["auc", "f1", "max_psi"],
        ["AUC por período", "F1 por período", "PSI máximo"],
    ):
        for window in ["expanding", "30d", "60d", "90d"]:
            part = df[df["window"] == window].sort_values("period")
            ax.plot(
                part["period"],
                part[metric],
                marker="o",
                linewidth=2,
                color=colors[window],
                label=labels[window],
            )
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_xlabel("Bloque futuro")
        ax.set_xticks(sorted(df["period"].unique()))
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Valor")
    axes[2].axhline(0.20, color="#8c2f39", linestyle="--", linewidth=1.3, label="Umbral PSI")
    axes[2].set_ylim(bottom=0)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Evaluación temporal adaptativa: XGBoost con features causales", fontsize=13, weight="bold")
    fig.savefig(FIGURES / "fig_adaptacion_metricas.pdf", bbox_inches="tight")
    plt.close(fig)


def arquitectura() -> None:
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.set_xlim(-0.45, 13.2)
    ax.set_ylim(0, 5)
    ax.axis("off")

    nodes = [
        (0.3, 2.0, 1.45, 1.0, "Datos\ntransacción + identidad", "#dceef2"),
        (2.1, 2.0, 1.55, 1.0, "Validación\ntemporal", "#d9e8fb"),
        (4.05, 2.0, 1.65, 1.0, "Features\ncausales + UID", "#e7e0f8"),
        (6.1, 2.0, 1.55, 1.0, "Modelo\nXGBoost", "#fbe4d5"),
        (8.05, 2.0, 1.65, 1.0, "Incertidumbre\ny score", "#f9f0c7"),
        (10.1, 2.0, 1.75, 1.0, "Decisión\naprobar / revisar / bloquear", "#dcebd8"),
        (6.15, 0.45, 2.0, 0.9, "Monitoreo\nAUC · PSI · costo", "#f2d7df"),
        (8.8, 0.45, 2.2, 0.9, "Adaptación\nventana + reentrenamiento", "#f2d7df"),
    ]
    for x, y, w, h, text, color in nodes:
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.2,
            edgecolor="#283845",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, weight="bold", color="#283845")

    for start, end in [
        ((1.75, 2.5), (2.1, 2.5)),
        ((3.65, 2.5), (4.05, 2.5)),
        ((5.7, 2.5), (6.1, 2.5)),
        ((7.65, 2.5), (8.05, 2.5)),
        ((9.7, 2.5), (10.1, 2.5)),
        ((10.95, 2.0), (9.9, 1.35)),
        ((7.15, 2.0), (7.15, 1.35)),
        ((7.15, 1.35), (8.8, 0.9)),
        ((9.9, 1.35), (9.9, 1.35)),
    ]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.3, color="#536878"))

    ax.text(0.3, 4.35, "Sistema inteligente adaptativo para detección de fraude", fontsize=15, weight="bold", color="#173f5f")
    ax.text(0.3, 3.85, "El feedback de desempeño y drift decide cuándo actualizar el modelo", fontsize=10, color="#536878")
    fig.savefig(FIGURES / "fig_arquitectura_adaptativa.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    metricas()
    arquitectura()
    print("Figuras adaptativas generadas en", FIGURES)
