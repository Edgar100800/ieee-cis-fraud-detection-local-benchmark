"""Extrae codigo y markdown de los .ipynb descargados a archivos .py legibles."""
import json
import re
from pathlib import Path

NB_DIR = Path("codigos")
PY_DIR = Path("codigos")

TECHNIQUES = {
    "LightGBM": r"(lightgbm|lgb\.|LGBMClassifier)",
    "XGBoost": r"(xgboost|XGBClassifier|xgb\.)",
    "CatBoost": r"(catboost|CatBoostClassifier)",
    "GPU (device=cuda/gpu)": r"(device.*=.*['\"](?:cuda|gpu)|tree_method.*gpu|gpu_hist|task_type.*GPU)",
    "SMOTE / oversampling": r"(SMOTE|imblearn|RandomOverSampler|scale_pos_weight|is_unbalance)",
    "Validacion temporal": r"(TimeSeriesSplit|GroupTimeSeriesSplit|time.*split.*valid)",
    "KFold / StratifiedKFold": r"(StratifiedKFold|KFold\(|GroupKFold)",
    "Hyperopt/Optuna/Bayes": r"(hyperopt|fmin|Trials|optuna|BayesianOptimization|bayes_opt)",
    "Feature agregacion (groupby)": r"(groupby\(.*\)\[.*\]\.(?:mean|std|transform|agg))",
    "UID / magic client": r"(uid|UID|client.*identif|card1.*addr1)",
    "Normalizacion D-cols": r"(D\d\s*-\s*D\d|D\d.*normalize|/ *D15)",
    "Label encoding": r"(LabelEncoder)",
    "Frequency encoding": r"(value_counts\(\)|freq_map|frequency_encode)",
    "Log transform amount": r"(np\.log1?\(?[^)]*TransactionAmt|TransactionAmt.*log)",
    "Adversarial validation": r"(adversarial)",
    "Reduccion de memoria": r"(reduce_mem|memory_usage.*astype|float32)",
    "Pseudo-labeling": r"(pseudo|Pseudo)",
    "Ensemble/blend": r"(blend|ensemble|0\.\d+\s*\*\s*sub|rank_avg)",
    "EDA con plotly": r"(plotly|go\.Scatter|px\.scatter|cufflinks)",
    "EDA con matplotlib/seaborn": r"(matplotlib|seaborn|plt\.|sns\.)",
}


def extract_code(ipynb: Path) -> tuple[str, str, dict]:
    nb = json.loads(ipynb.read_text())
    code_parts, md_parts = [], []
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "code":
            code_parts.append(f"# %% [{cell.get('execution_count', '?')}]\n{src}\n")
        elif cell["cell_type"] == "markdown":
            heads = re.findall(r"^#+\s+(.+)$", src, re.M)
            md_parts.extend(heads)
    full_code = "\n".join(code_parts)
    summary = {
        "titulo": nb.get("metadata", {}).get("kernelspec", {}).get("display_name", ipynb.stem),
        "secciones_md": md_parts,
        "n_celdas_code": len(code_parts),
        "tecnicas": {},
    }
    for tech, pat in TECHNIQUES.items():
        hits = len(re.findall(pat, full_code, re.I))
        if hits:
            summary["tecnicas"][tech] = hits
    return full_code, "\n".join(f"- {h}" for h in md_parts), summary


def main() -> None:
    PY_DIR.mkdir(exist_ok=True)
    all_summaries = []
    for ipynb in sorted(NB_DIR.glob("*.ipynb")):
        code, md_headers, summary = extract_code(ipynb)
        out = PY_DIR / f"{ipynb.stem}.py"
        header = f'"""\nNOTEBOOK: {ipynb.stem}\n\nSECCIONES (markdown del notebook):\n{md_headers}\n\nTECNICAS DETECTADAS:\n'
        for k, v in summary["tecnicas"].items():
            header += f"  - {k} ({v} menciones)\n"
        header += '"""\n\n'
        out.write_text(header + code)
        all_summaries.append((ipynb.stem, summary))
        print(f"extraido: {out}")

    report = ["# Resumen de tecnicas por notebook\n"]
    for stem, s in all_summaries:
        report.append(f"\n## {stem}")
        report.append(f"- Celdas de codigo: {s['n_celdas_code']}")
        report.append(f"- Seciones: {', '.join(s['secciones_md'][:15]) or '-'}")
        report.append("- Tecnicas: " + (", ".join(f"{k}({v})" for k, v in s["tecnicas"].items()) or "-"))
    (PY_DIR / "_RESUMEN_TECNICAS.md").write_text("\n".join(report))
    print(f"\nresumen -> {PY_DIR/'_RESUMEN_TECNICAS.md'}")


if __name__ == "__main__":
    main()
