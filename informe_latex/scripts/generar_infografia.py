"""Genera una infografia de una pagina para la entrega del proyecto."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "informe_latex" / "figures" / "infografia_sistema_adaptativo.pdf"


BG = "#F7F8FA"
INK = "#17212B"
MUTED = "#52606D"
BLUE = "#1769AA"
BLUE_LIGHT = "#DCEEFF"
GREEN = "#138A72"
GREEN_LIGHT = "#DDF5EF"
ORANGE = "#D97706"
ORANGE_LIGHT = "#FFF0D8"
RED = "#B42318"
RED_LIGHT = "#FDE7E5"
LINE = "#CBD5E1"


def box(ax, x, y, w, h, title, body, face, edge, title_color=INK, fontsize=8.2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.4, edgecolor=edge, facecolor=face,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(x + 0.018, y + h - 0.035, title, transform=ax.transAxes,
            fontsize=10, fontweight="bold", color=title_color, va="top")
    ax.text(x + 0.018, y + h - 0.075, body, transform=ax.transAxes,
            fontsize=fontsize, color=INK, va="top", linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color=BLUE):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), transform=ax.transAxes,
        arrowstyle="-|>", mutation_scale=13, linewidth=1.8,
        color=color, connectionstyle="arc3,rad=0",
    ))


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13.333, 7.5), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    # Header
    ax.text(0.045, 0.935, "SISTEMA INTELIGENTE ADAPTATIVO", fontsize=22,
            fontweight="bold", color=INK, va="top")
    ax.text(0.045, 0.895, "Deteccion y respuesta ante concept drift en fraude financiero IEEE-CIS",
            fontsize=11.5, color=MUTED, va="top")
    ax.text(0.955, 0.93, "UTEC · Proyecto #1", fontsize=9.5, color=BLUE,
            ha="right", va="top", fontweight="bold")

    # Evidence strip
    strip_y, strip_h = 0.79, 0.075
    stats = [
        ("590,540", "transacciones de entrenamiento", BLUE),
        ("3.5%", "tasa de fraude", RED),
        ("0.9479", "AUC del mejor baseline", GREEN),
        ("0.20", "umbral PSI de alerta", ORANGE),
    ]
    for i, (value, label, color) in enumerate(stats):
        x = 0.045 + i * 0.23
        ax.add_patch(FancyBboxPatch((x, strip_y), 0.205, strip_h,
                                    boxstyle="round,pad=0.009,rounding_size=0.012",
                                    facecolor="white", edgecolor=LINE, linewidth=0.8,
                                    transform=ax.transAxes))
        ax.text(x + 0.015, strip_y + 0.045, value, transform=ax.transAxes,
                fontsize=15, fontweight="bold", color=color, va="center")
        ax.text(x + 0.015, strip_y + 0.017, label, transform=ax.transAxes,
                fontsize=7.3, color=MUTED, va="center")

    # Main architecture flow
    ax.text(0.045, 0.745, "CICLO OPERATIVO", fontsize=10.5, fontweight="bold", color=INK)
    y, h, w = 0.555, 0.14, 0.145
    items = [
        ("1 · DATOS", "Transacciones + identidad\nordenadas por tiempo", BLUE_LIGHT, BLUE),
        ("2 · FEATURES", "Variables temporales\ny estadisticas causales", GREEN_LIGHT, GREEN),
        ("3 · MODELO", "XGBoost + baseline\nprobabilidad de fraude", ORANGE_LIGHT, ORANGE),
        ("4 · DECISION", "Umbral operativo\nalerta / revision", RED_LIGHT, RED),
        ("5 · MONITOREO", "PSI + metricas\npor periodo", BLUE_LIGHT, BLUE),
    ]
    xs = [0.045 + i * 0.19 for i in range(5)]
    for i, ((title, body, face, edge), x) in enumerate(zip(items, xs)):
        box(ax, x, y, w, h, title, body, face, edge)
        if i < len(items) - 1:
            arrow(ax, x + w + 0.008, y + h / 2, xs[i + 1] - 0.01, y + h / 2)
    # Monitoring loop
    arrow(ax, xs[-1] + w / 2, y - 0.01, xs[1] + w / 2, y - 0.08, color=ORANGE)
    ax.text(0.66, 0.46, "si cambia el entorno → ajustar ventana y reentrenar",
            fontsize=8.4, color=ORANGE, fontweight="bold", ha="center")

    # Evidence / adaptation blocks
    box(ax, 0.045, 0.235, 0.285, 0.17, "PROTOCOLO TEMPORAL",
        "Entrenamiento: pasado\nEvaluacion: bloques futuros\nVentanas: expansiva, 30, 60 y 90 dias",
        "white", LINE, fontsize=8.0)
    box(ax, 0.36, 0.235, 0.285, 0.17, "RESULTADO ADAPTATIVO",
        "XGBoost · ventana de 90 dias\nAUC 0.9156 · F1 0.5635\nCosto medio de decision: 17,336",
        GREEN_LIGHT, GREEN, fontsize=8.0)
    box(ax, 0.675, 0.235, 0.28, 0.17, "LECTURA CRITICA",
        "El PSI no supero 0.20 en los bloques\nevaluados, pero el rendimiento fluctua.\nSe requiere monitoreo continuo.",
        ORANGE_LIGHT, ORANGE, fontsize=8.0)

    # Footer conclusion
    ax.add_patch(FancyBboxPatch((0.045, 0.08), 0.91, 0.095,
                                boxstyle="round,pad=0.012,rounding_size=0.015",
                                facecolor=INK, edgecolor=INK, transform=ax.transAxes))
    ax.text(0.065, 0.135, "CONCLUSION", transform=ax.transAxes, fontsize=9,
            fontweight="bold", color="#A7D8FF", va="center")
    ax.text(0.19, 0.135,
            "La adaptacion combina ventanas deslizantes, features causales y decisiones con costo explicito para sostener el modelo en el tiempo.",
            transform=ax.transAxes, fontsize=10.2, color="white", va="center")
    ax.text(0.955, 0.035, "Fuente: desarrollo propio sobre IEEE-CIS Fraud Detection",
            transform=ax.transAxes, fontsize=7.2, color=MUTED, ha="right")

    fig.savefig(OUT, format="pdf", bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
