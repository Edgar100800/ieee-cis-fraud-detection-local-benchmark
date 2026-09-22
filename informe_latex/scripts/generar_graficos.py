"""Genera todos los graficos (PDF vectorial) y tablas LaTeX del informe.

Uso: .venv/bin/python informe_latex/scripts/generar_graficos.py
Salidas: informe_latex/figures/*.pdf e informe_latex/tables/*.tex
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "informe_latex" / "figures"
TABLES = ROOT / "informe_latex" / "tables"
FIGS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
AZUL, ROJO, VERDE, GRIS = "#1f77b4", "#d62728", "#2ca02c", "#7f7f7f"

# ---------------------------------------------------------------- datos
print("Cargando train_transaction.csv ...")
tt = pd.read_csv(
    ROOT / "data/raw/train_transaction.csv",
    index_col="TransactionID",
)
ti = pd.read_csv(ROOT / "data/raw/train_identity.csv", index_col="TransactionID")
y = tt["isFraud"]
print("OK", tt.shape)

# ------------------------------------------------ 1. distribucion del target
fig, ax = plt.subplots(figsize=(5.2, 3.2))
counts = y.value_counts().sort_index()
bars = ax.bar(["Legítima (0)", "Fraude (1)"], counts.values, color=[AZUL, ROJO], width=0.55)
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 8000, f"{v:,}", ha="center", fontsize=10)
ax.set_ylabel("Número de transacciones")
ax.set_title(f"Distribución del target — fraude: {y.mean():.2%} (razón 1:{int(1/y.mean())})")
ax.set_ylim(0, counts.max() * 1.12)
fig.tight_layout()
fig.savefig(FIGS / "fig_distribucion_target.pdf")
plt.close(fig)

# ------------------------------------------------ 2. nulos por familia
def familia(col):
    if col.startswith("V"):
        return "V1–V339"
    if col.startswith("C"):
        return "C1–C14"
    if col.startswith("D"):
        return "D1–D15"
    if col.startswith("M"):
        return "M1–M9"
    if col.startswith("id_") or col in ("DeviceType", "DeviceInfo"):
        return "identity"
    if col.startswith("card"):
        return "card1–6"
    if col.startswith(("addr", "dist")):
        return "addr/dist"
    if "emaildomain" in col:
        return "emails"
    return "otras"

grp = tt.drop(columns="isFraud").T.groupby(pd.Series(tt.drop(columns="isFraud").columns.map(familia)).values)
nulos = (grp.apply(lambda df: df.isna().mean().mean()) * 100).sort_values()
presencia = ti.notna().mean().mean() * 100

fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.barh(nulos.index, nulos.values, color=AZUL)
for i, v in enumerate(nulos.values):
    ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=9)
ax.set_xlabel("% de valores faltantes (promedio de la familia)")
ax.set_title("Valores faltantes por familia de features\n(identity presente solo en "
             f"{presencia:.0f}% de las transacciones)")
ax.set_xlim(0, 100)
fig.tight_layout()
fig.savefig(FIGS / "fig_nulos_por_grupo.pdf")
plt.close(fig)

# ------------------------------------------------ 3. distribucion del monto
fig, ax = plt.subplots(figsize=(6.0, 3.4))
for cls, c, lbl in [(0, AZUL, "Legítima"), (1, ROJO, "Fraude")]:
    vals = np.log10(tt.loc[y == cls, "TransactionAmt"] + 1)
    ax.hist(vals, bins=60, alpha=0.6, density=True, color=c, label=lbl)
ax.set_xlabel(r"$\log_{10}(\mathrm{TransactionAmt}+1)$")
ax.set_ylabel("densidad")
ax.set_title("Distribución del monto por clase (cola pesada → justifica log1p)")
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "fig_amt_distribucion.pdf")
plt.close(fig)

# ------------------------------------------------ 4. fraude por ProductCD
rate = tt.groupby("ProductCD")["isFraud"].agg(["mean", "size"]).sort_values("mean")
fig, ax = plt.subplots(figsize=(5.6, 3.2))
bars = ax.bar(rate.index, rate["mean"] * 100, color=[ROJO if v > y.mean() * 100 else AZUL for v in rate["mean"]])
ax.axhline(y.mean() * 100, color=GRIS, ls="--", lw=1, label=f"tasa global {y.mean():.1%}")
for b, v in zip(bars, rate["mean"] * 100):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.1f}%", ha="center", fontsize=9)
ax.set_xlabel("ProductCD")
ax.set_ylabel("% de fraude")
ax.set_title("Tasa de fraude por tipo de producto")
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "fig_fraude_productcd.pdf")
plt.close(fig)

# ------------------------------------------------ 5. fraude en el tiempo
day = tt["TransactionDT"] / 86400.0
daily = y.groupby(day.round().astype(int)).agg(["mean", "size"])
daily = daily[daily["size"] > 500]
fig, ax = plt.subplots(figsize=(7.2, 3.2))
ax.plot(daily.index, daily["mean"] * 100, lw=0.9, color=ROJO)
ax.axhline(y.mean() * 100, color=GRIS, ls="--", lw=1, label=f"tasa global {y.mean():.1%}")
ax.set_xlabel("día relativo (TransactionDT/86400)")
ax.set_ylabel("% de fraude diario")
ax.set_title("Tasa de fraude a lo largo del tiempo: picos cíclicos (campañas de ataque)")
ax.legend()
fig.tight_layout()
fig.savefig(FIGS / "fig_fraude_temporal.pdf")
plt.close(fig)

# ------------------------------------------------ 6. normalizacion D15
d15n = tt["D15"] - tt["TransactionDT"] / 86400.0
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), sharey=True)
axes[0].scatter(day, tt["D15"], s=0.3, alpha=0.15, color=AZUL)
axes[0].set_title(r"D15 original — deriva con el tiempo")
axes[0].set_xlabel("día relativo")
axes[0].set_ylabel("D15 [días]")
axes[1].scatter(day, d15n, s=0.3, alpha=0.15, color=VERDE)
axes[1].set_title(r"D15 normalizado ($D15 - DT/86400$) — banda estable")
axes[1].set_xlabel("día relativo")
axes[1].set_ylim(-40, 40)
fig.suptitle("Normalización de columnas D (idea de Deotte): features comparables en el futuro", y=1.02)
fig.tight_layout()
fig.savefig(FIGS / "fig_d15_normalizacion.pdf", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------ 7. esquema del UID magico
fig, ax = plt.subplots(figsize=(8.6, 3.6))
ax.axis("off")
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)

def caja(x, yv, w, h, txt, fc):
    ax.add_patch(plt.Rectangle((x, yv), w, h, fc=fc, ec="black", lw=1.1, zorder=2))
    ax.text(x + w / 2, yv + h / 2, txt, ha="center", va="center", fontsize=9, zorder=3)

def flecha(x1, y1, x2, y2, txt=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.3, color="black"))
    if txt:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, txt, ha="center", fontsize=8, style="italic")

caja(0.1, 2.4, 2.2, 1.0, "card1 + addr1\n(tarjeta + dirección)", "#dbe9f6")
caja(0.1, 1.0, 2.2, 1.0, "floor(día − D1)\n(fecha alta cuenta)", "#dbe9f6")
caja(3.1, 1.7, 2.0, 1.1, "UID\n(pseudo-cliente)", "#ffe9a8")
caja(6.0, 3.0, 3.7, 0.8, "Frecuencia del grupo (FE)\n¿cuántas tx hace el cliente?", "#d5f0d5")
caja(6.0, 2.0, 3.7, 0.8, "mean / std por grupo (AG)\n¿es anómalo vs su historial?", "#d5f0d5")
caja(6.0, 1.0, 3.7, 0.8, "nunique por grupo (AG2)\n¿cuántos valores distintos usa?", "#d5f0d5")
caja(6.0, 0.1, 3.7, 0.7, "El UID se ELIMINA: nunca entra al modelo", "#f2f2f2")
flecha(2.3, 2.9, 3.1, 2.55)
flecha(2.3, 1.5, 3.1, 1.95)
flecha(5.1, 2.55, 6.0, 3.4)
flecha(5.1, 2.25, 6.0, 2.4)
flecha(5.1, 1.95, 6.0, 1.4)
flecha(7.8, 3.0, 7.8, 2.85)
flecha(7.8, 2.0, 7.8, 1.8)
flecha(7.8, 1.0, 7.8, 0.85)
ax.set_title("Técnica del 'magic UID': reconstruir al cliente y agregar su historial", fontsize=11)
fig.tight_layout()
fig.savefig(FIGS / "fig_esquema_uid.pdf")
plt.close(fig)

# ------------------------------------------------ 8-10. resultados de benchmarks
r1 = pd.read_csv(ROOT / "results/bench_01_inversion.csv").iloc[0]
r2 = pd.read_csv(ROOT / "results/bench_02_xhlulu.csv").iloc[0]
r3 = pd.read_csv(ROOT / "results/bench_03_nroman.csv").iloc[0]
r4 = pd.read_csv(ROOT / "results/bench_04_cdeotte.csv")
b95 = r4[r4.notebook == "cdeotte-xgb95-sin-magic"].iloc[0]
b96 = r4[r4.notebook == "cdeotte-xgb96-magic"].iloc[0]

modelos = [
    ("Baseline XGB CPU\n(inversion)", float(r1.auc_holdout), float(r1.fit_s), AZUL, "holdout 75/25"),
    ("Baseline XGB GPU\n(xhlulu)", float(r2.auc_holdout), float(r2.fit_s), AZUL, "holdout 75/25"),
    ("LightGBM + FE\n(nroman)", float(r3.auc_mean_timeseriessplit), float(r3.cv_s), VERDE, "TimeSeriesSplit ×3"),
    ("XGB d12 sin UID\n(cdeotte 95)", float(b95.auc_oof_groupkfold3), float(b95.holdout_s + b95.cv_s), ROJO, "GroupKFold mes ×3"),
    ("XGB d12 + magic UID\n(cdeotte 96)", float(b96.auc_oof_groupkfold3), float(b96.holdout_s + b96.cv_s), ROJO, "GroupKFold mes ×3"),
]
names = [m[0] for m in modelos]
aucs = [m[1] for m in modelos]
times = [m[2] for m in modelos]
cols = [m[3] for m in modelos]

fig, ax = plt.subplots(figsize=(7.6, 3.8))
bars = ax.barh(names, aucs, color=cols)
ax.set_xlim(0.915, 0.952)
for b, v in zip(bars, aucs):
    ax.text(v + 0.0004, b.get_y() + b.get_height() / 2, f"{v:.4f}", va="center", fontsize=9)
ax.axvline(aucs[0], color=GRIS, ls=":", lw=1)
ax.set_xlabel("AUC en validación local (temporal)")
ax.set_title("Comparación de AUC local — mayor es mejor")
fig.tight_layout()
fig.savefig(FIGS / "fig_auc_comparacion.pdf")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.scatter(times, aucs, s=140, c=cols, zorder=3)
for m in modelos:
    ax.annotate(m[0].replace("\n", " "), (m[2], m[1]), textcoords="offset points",
                xytext=(8, -12), fontsize=8)
ax.set_xlabel("tiempo de cómputo [s] (escala log)")
ax.set_xscale("log")
ax.set_ylabel("AUC local")
ax.set_title("Precisión vs costo computacional (16 cores CPU / RTX 2060 6 GB)")
fig.tight_layout()
fig.savefig(FIGS / "fig_auc_vs_tiempo.pdf")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.2, 3.4))
xpos = [0, 1]
hold = [float(b95.auc_holdout), float(b96.auc_holdout)]
oof = [float(b95.auc_oof_groupkfold3), float(b96.auc_oof_groupkfold3)]
w = 0.36
ax.bar([p - w / 2 for p in xpos], hold, width=w, color=GRIS, label="holdout 75/25")
ax.bar([p + w / 2 for p in xpos], oof, width=w, color=ROJO, label="OOF GroupKFold 3 meses")
for i, (hv, ov) in enumerate(zip(hold, oof)):
    ax.text(i - w / 2, hv + 0.0004, f"{hv:.4f}", ha="center", fontsize=9)
    ax.text(i + w / 2, ov + 0.0004, f"{ov:.4f}", ha="center", fontsize=9)
ax.set_xticks(xpos)
ax.set_xticklabels(["XGB_95 (sin UID)", "XGB_96 (con magic UID)"])
ax.set_ylim(0.93, 0.951)
ax.set_ylabel("AUC")
ax.set_title(f"Ablación: efecto del magic UID = +{oof[1]-oof[0]:.4f} AUC (mismo modelo d12)")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIGS / "fig_ablacion_uid.pdf")
plt.close(fig)

# ------------------------------------------------ 11. leaderboard publico
lb_path = next((ROOT / "data/raw").glob("*publicleaderboard*.csv"))
with open(lb_path, encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
scores = np.array([float(r["Score"]) for r in rows])
equipos = [r["TeamName"] for r in rows]
orden = np.argsort(-scores)
fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.hist(scores, bins=80, color=AZUL, alpha=0.75)
for k in range(5):
    i = orden[k]
    ax.axvline(scores[i], color=ROJO, lw=1.2)
    ax.text(scores[i], ax.get_ylim()[1] * 0.92 - k * 28, f"{k+1}. {equipos[i][:14]}",
            rotation=90, fontsize=7.5, va="top", color=ROJO)
ax.set_xlabel("score público (AUC)")
ax.set_ylabel("nº de equipos")
ax.set_title(f"Leaderboard público IEEE-CIS: {len(scores):,} equipos — top 5 en rojo")
fig.tight_layout()
fig.savefig(FIGS / "fig_leaderboard.pdf")
plt.close(fig)

# ------------------------------------------------ 12. importancia de features (fit rapido GPU)
print("Ajustando XGB rapido para importancia de features ...")
import xgboost as xgb
from sklearn.metrics import roc_auc_score

vcols = [c for c in tt.columns if c.startswith(("V", "C", "D"))][:170] + \
    ["TransactionAmt", "TransactionDT", "card1", "card2", "card5", "addr1", "addr2", "dist1", "P_emaildomain", "ProductCD"]
Xq = tt[vcols].copy()
for c in Xq.columns:
    if not pd.api.types.is_numeric_dtype(Xq[c]):
        codes, _ = pd.factorize(Xq[c].astype(str), sort=True)
        Xq[c] = codes.astype(np.int32)
Xq = Xq.fillna(-1).astype(np.float32)
m = xgb.XGBClassifier(n_estimators=150, max_depth=9, learning_rate=0.1,
                      tree_method="hist", device="cuda", eval_metric="auc", n_jobs=16)
m.fit(Xq, y)
imp = pd.Series(m.feature_importances_, index=Xq.columns).nlargest(20)[::-1]
auc_rapido = roc_auc_score(y, m.predict_proba(Xq)[:, 1])
print(f"AUC in-sample del fit rapido: {auc_rapido:.4f} (solo orientativo)")
del Xq, m

fig, ax = plt.subplots(figsize=(6.4, 4.6))
ax.barh(imp.index, imp.values, color=AZUL)
ax.set_xlabel("importancia (gain, normalizada)")
ax.set_title("Top-20 features — XGB rápido sobre train completo\n(Las C/D y card1+addr1 dominan: señal de cliente)")
fig.tight_layout()
fig.savefig(FIGS / "fig_top_features.pdf")
plt.close(fig)

# ------------------------------------------------ tablas LaTeX
def tabla_ranking():
    filas = [
        ("Baseline XGBoost CPU (inversion)", f"{float(r1.auc_holdout):.4f}", f"{float(r1.fit_s):.0f}\\,s", "holdout temporal 75/25", "432 features crudas"),
        ("Baseline XGBoost GPU (xhlulu)", f"{float(r2.auc_holdout):.4f}", f"{float(r2.fit_s):.0f}\\,s", "holdout temporal 75/25", "identico al anterior, en GPU"),
        ("LightGBM + FE (nroman)", f"{float(r3.auc_mean_timeseriessplit):.4f}", f"{float(r3.cv_s):.0f}\\,s", "TimeSeriesSplit $\\times$3", "RFE 120 features + count encoding"),
        ("XGBoost d12 sin UID (cdeotte 95)", f"{float(b95.auc_oof_groupkfold3):.4f}", f"{float(b95.holdout_s + b95.cv_s):.0f}\\,s", "GroupKFold mensual $\\times$3", "V-cols seleccionadas + D normalizadas"),
        ("XGBoost d12 + magic UID (cdeotte 96)", f"\\textbf{{{float(b96.auc_oof_groupkfold3):.4f}}}", f"{float(b96.holdout_s + b96.cv_s):.0f}\\,s", "GroupKFold mensual $\\times$3", "UID + $\\sim$40 agregaciones por cliente"),
    ]
    lines = [
        "\\begin{tabular}{lrccl}", "\\toprule",
        "Modelo & AUC & Tiempo & Validación & Técnica clave \\\\", "\\midrule",
    ]
    lines += [f"{a} & {b} & {c} & {d} & {e} \\\\" for a, b, c, d, e in filas]
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / "tabla_ranking.tex").write_text("\n".join(lines))

def tabla_datos():
    n_train, n_cols = tt.shape
    n_id = ti.shape[0]
    filas = [
        ("train\\_transaction.csv", f"{n_train:,}", f"{n_cols}", "transacciones + 339 V + conteos + tiempos"),
        ("train\\_identity.csv", f"{n_id:,}", f"{ti.shape[1]}", f"dispositivo / red (presente en {ti.shape[0]/n_train*100:.0f}\\% de las filas)"),
        ("test\\_transaction.csv", "506{,}691", "393", "mes posterior al train (sin target)"),
        ("Fraude (isFraud=1)", f"{int(y.sum()):,}", "---", f"{y.mean()*100:.2f}\\%"),
    ]
    lines = ["\\begin{tabular}{lrll}", "\\toprule",
             "Archivo / magnitud & Filas & Columnas & Nota \\\\", "\\midrule"]
    lines += [f"{a} & {b} & {c} & {d} \\\\" for a, b, c, d in filas]
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TABLES / "tabla_datos.tex").write_text("\n".join(lines))

tabla_ranking()
tabla_datos()
print("Tablas LaTeX escritas en", TABLES)
print("Listo: figuras en", FIGS)
