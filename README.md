# Bank Marketing Campaign Analysis, Customer Segmentation and Subscription Prediction Using Python

## Current data status

The supplied path is `data/bank_marketing.csv`, but inspection on 2026-10-02 found that it is 0 bytes. It contains no header or observations. Therefore dataset format/version/source, row and column counts, target values, cleaning counts, EDA findings, cluster solution, and model metrics cannot currently be established. No results or conclusions are fabricated. The analysis command intentionally stops with a clear error until a real delimited dataset is supplied.

No raw dataset is overwritten. Place the verified source CSV at `data/bank_marketing.csv`, or pass its path with `--data`. The target defaults to `y`; specify the actual target with `--target` if different. The project does not claim Kaggle or UCI provenance based only on a filename.

## Project question

Banks use marketing campaigns to encourage customers to subscribe to financial products. This project investigates which customer and campaign characteristics are associated with the observed subscription outcome, whether available numeric characteristics support useful customer segmentation, and how supervised models compare when predicting subscriptions. Any recommendations must be based on the generated results and campaign context, not treated as causal claims.

## Objectives

- Inspect schema, data types, missing and `unknown` values, duplicates, unique values, and target distribution.
- Clean exact duplicate records while reporting unknown and missing values rather than silently recoding them.
- Explore demographics and campaign variables with generated summaries and plots.
- Add age/contact-history features only when supporting source columns exist.
- Compare numeric K-means cluster counts using a silhouette score.
- Compare logistic regression and random forest on a stratified holdout using accuracy, precision, recall, F1, and ROC-AUC.
- Keep preprocessing inside the training pipeline and exclude `duration` from predictive features to reduce post-contact leakage for a pre-contact targeting use case.
- Record limitations, provenance, actual results, and evidence-based recommendations after the source data is verified.

## Project structure

```text
Bank_Marketing_Capstone/
├── data/
│   ├── bank_marketing.csv       # Existing supplied path; currently empty
│   └── processed/               # Generated cleaned data
├── notebooks/
│   └── bank_marketing_analysis.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── clustering.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── analysis.py
├── visualizations/              # Generated figures and profile
├── models/                      # Selected serialized classifier
├── reports/                     # Generated summary and final report scaffold
├── report/                      # Existing supplied report directory, preserved
├── requirements.txt
└── README.md
```

## Reproduce

Use Python 3.10 or later. From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.analysis --data data/bank_marketing.csv --target y
```

Open and run `notebooks/bank_marketing_analysis.ipynb` from the project root or set its project root cell appropriately. The notebook calls the same pipeline as the CLI.

## Outputs

- `data/processed/bank_marketing_cleaned.csv`: duplicate-cleaned data with conditional engineered fields.
- `visualizations/data_profile.json` and PNG plots: data-derived quality summary and EDA visuals.
- `reports/analysis_summary.json`: actual quality, profile, segmentation, and model results.
- `reports/customer_segments.csv` and `reports/kmeans.joblib`: segment assignments and fitted clustering objects when calculation is possible.
- `models/best_model.joblib`: model selected by holdout F1, with ROC-AUC as tie-breaker.

Generated files are excluded from version control by `.gitignore`; the source file is not.

## Method and interpretation

The loader supports comma, semicolon, tab, and pipe delimiters and normalizes field names. It reports missing and literal `unknown` values separately. Numeric variables use median imputation and scaling; categorical fields use most-frequent imputation and one-hot encoding. The train/test split is stratified with a fixed random seed. Metrics are holdout estimates, not guarantees of future campaign performance. Class balancing is enabled for both model candidates. The choice to exclude `duration` reflects deployment timing: call duration is not known before placing a call. Segmentation is restricted to nonconstant numeric predictors and does not use the target or duration.

## Data provenance and limitations

The local file is empty, so its origin cannot be verified. After obtaining the data, document the exact source URL or repository, dataset/version name, access date, license/terms, and any transformations. If the dataset is the UCI Bank Marketing collection, cite the exact variant and original UCI Machine Learning Repository record; if from Kaggle, identify the exact dataset page and author. Do not infer a source solely from familiar column names.

Model associations do not establish causality. Results can be affected by class imbalance, sampling, campaign period, customer privacy constraints, and distribution shift. If the dataset is the common Portuguese bank telemarketing dataset, its campaign-specific context and the `duration` leakage risk should be discussed explicitly.
