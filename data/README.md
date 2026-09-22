# data/

Esta carpeta **no incluye los datos** (1.4 GB comprimidos, licencia de uso de la
competencia). Descargarlos con:

```bash
kaggle competitions download -c ieee-fraud-detection -p data/raw
unzip 'data/raw/ieee-fraud-detection.zip' -d data/raw
```

Requiere [cuenta de Kaggle](https://www.kaggle.com/settings) (API token en
`~/.kaggle/kaggle.json`) y haber aceptado las reglas de la competencia en
https://www.kaggle.com/competitions/ieee-fraud-detection/rules

Archivos esperados en `data/raw/`:

| Archivo | Tamaño |
|---|---:|
| train_transaction.csv | 683 MB |
| test_transaction.csv | 613 MB |
| train_identity.csv | 26 MB |
| test_identity.csv | 26 MB |
| sample_submission.csv | 6 MB |
