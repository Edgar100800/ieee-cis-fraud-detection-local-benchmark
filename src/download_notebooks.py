"""Descarga los notebooks top de IEEE-CIS Fraud Detection sin autenticacion."""
import json
import time
import urllib.request

NOTEBOOKS = [
    "artgor/eda-and-models",
    "kabure/extensive-eda-and-modeling-xgb-hyperopt",
    "cdeotte/xgb-fraud-with-magic-0-9600",
    "jesucristo/fraud-complete-eda",
    "shahules/tackling-class-imbalance",
    "robikscube/ieee-fraud-detection-first-look-and-eda",
    "nroman/lgb-single-model-lb-0-9419",
    "kyakovlev/ieee-gb-2-make-amount-useful-again",
    "xhlulu/ieee-fraud-xgboost-with-gpu-fit-in-40s",
    "vincentlugat/ieee-lgb-bayesian-opt",
    "inversion/ieee-simple-xgboost",
    "cdeotte/rapids-feature-engineering-fraud-0-96",
]

OUT_DIR = "codigos"


def save_nb(slug: str, payload: dict) -> str:
    meta = payload["metadata"]
    src = payload["blob"]["source"]
    if isinstance(src, list):
        src = "".join(src)
    nb = json.loads(src)
    fname = f"{OUT_DIR}/{slug.replace('/', '__')}.ipynb"
    with open(fname, "w") as f:
        json.dump(nb, f, indent=1)
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    n_md = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
    return (
        f"OK {slug:55s} votes={meta.get('totalVotes'):>5} "
        f"cells(code/md)={n_code}/{n_md} -> {fname}"
    )


def main() -> None:
    for slug in NOTEBOOKS:
        url = f"https://www.kaggle.com/api/v1/kernels/pull/{slug}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                payload = json.load(r)
            print(save_nb(slug, payload))
        except Exception as exc:
            print(f"ERR {slug}: {exc}")
        time.sleep(1)


if __name__ == "__main__":
    main()
