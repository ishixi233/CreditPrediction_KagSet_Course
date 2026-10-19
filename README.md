# Credit Card Default Prediction: Comparing Classification Models and Decision Thresholds

Group project for **MH6805 Machine Learning in Finance**, Nanyang Technological University.
Members: **Shi Xi, Li Erteng, Jiang Chengrui.**

We compare six classifiers for next-month credit-card default on the UCI *Default of Credit Card Clients* data (30,000 Taiwanese cardholders, 22.1% default) and study how the choice of decision threshold, the assumed cost ratio between missed defaults and false alarms, and probability calibration change which model a lender should operate.

## Deliverables

| Item | Location |
|---|---|
| Final report (PDF, 7 pages + appendices) | [report/Credit_Default_Report.pdf](report/Credit_Default_Report.pdf) |
| Report source (LaTeX, NeurIPS template) and Markdown mirror | [report/Credit_Default_Report.tex](report/Credit_Default_Report.tex), [report/Credit_Default_Report.md](report/Credit_Default_Report.md) |
| Proposal (as submitted) | [Proposal/LaTeX/Credit_Default_Proposal.pdf](Proposal/LaTeX/Credit_Default_Proposal.pdf), [Proposal/02_Credit_Default_Proposal.md](Proposal/02_Credit_Default_Proposal.md) |
| Code | `src/` (library), `scripts/01`–`10` (pipeline), `tests/` |
| Data | `default of credit card clients.xls` (raw, SHA-256 recorded in `src/config.py`), `data/splits/splits.csv` (shared 70/30 split and five folds) |
| Frozen configuration | [configs/frozen.json](configs/frozen.json): selected hyper-parameters and the cost-minimising threshold for each model and cost ratio, with the commit at which they were frozen |
| Results | `results/predictions/` (out-of-fold and test scores), `results/tables/`, `results/figures/`, `results/experiments/` (every cross-validation run) |
| Fitted models | `models/*.joblib` |

## Main findings

| Model | Test AP | ROC-AUC | Brier | Cost per client, r = 5 |
|---|---|---|---|---|
| Random Forest | **0.561** | 0.781 | 0.134 | **0.553** |
| AdaBoost | 0.539 | 0.765 | 0.139 | 0.569 |
| Logistic Regression | 0.538 | 0.765 | 0.136 | 0.587 |
| KNN | 0.521 | 0.759 | 0.141 | 0.590 |
| Decision Tree | 0.508 | 0.751 | 0.138 | 0.591 |
| Gaussian NB | 0.497 | 0.734 | 0.288 | 0.599 |

- Random Forest ranks defaulters best (paired bootstrap intervals exclude every competitor) and has the lowest weighted cost for cost ratios r ≥ 3; for r ≤ 2 it is statistically tied with AdaBoost, Logistic Regression and the Decision Tree.
- Threshold choice matters more than model choice: at the default threshold 0.5 every model is more expensive than the trivial policy of flagging everyone when r = 5; thresholds selected out-of-fold for an explicit cost ratio reduce cost by 23–29%.
- Calibration decides whether thresholds can be read as probabilities: Logistic Regression, Random Forest and the Decision Tree are close to calibrated; Gaussian NB ranks acceptably but its probabilities are worse than the base rate.

## Protocol

1. Stratified 70/30 split (seed 42); rows with identical predictors are kept in one partition and one fold. Five stratified folds inside the development set are stored once and shared by every model.
2. Preprocessing is the first step of each pipeline and is fitted only on the training part of each fold.
3. Hyper-parameters are chosen by mean five-fold average precision from small grids written before the experiments.
4. For each model and each cost ratio r ∈ {1, 2, 3, 5, 10}, the alert threshold minimising r·FN + FP is chosen on the pooled out-of-fold predictions (5:1 is the pre-specified primary scenario).
5. Parameters and thresholds are frozen in `configs/frozen.json`; the models are refitted on the full development set and the test set is scored **once**. `scripts/07_run_test.py` refuses to run without the frozen file or when test predictions already exist.
6. Uncertainty: paired stratified bootstrap of the test set (2,000 resamples).

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # versions are pinned; results differ slightly under other versions
python -m pytest                       # split integrity and metric functions
python scripts/01_check_data.py        # data checks -> docs/data_check.md
python scripts/03_run_cv.py            # cross-validation for all models -> results/experiments, results/predictions/oof_*
python scripts/04_threshold_cost_analysis.py
python scripts/05_calibration.py
python scripts/08_feature_interpretation.py
python scripts/09_report_figures.py
python scripts/10_test_uncertainty.py
```

`scripts/02_make_splits.py` and `scripts/06_freeze.py` regenerate the committed split and frozen configuration; `scripts/07_run_test.py` produced the committed test outputs and should not be run again. The report is built with `cd report && latexmk -pdf Credit_Default_Report.tex`.

Data documentation: [docs/data_dictionary.md](docs/data_dictionary.md) (variables, recoding, split sizes) and [docs/data_check.md](docs/data_check.md) (generated checks on the raw file).

## Layout

```
Proposal/      proposal as submitted (Markdown and LaTeX)
src/           config, data loading, preprocessing, metrics, cross-validation, model registry, plotting style
scripts/       numbered pipeline, 01 data checks ... 10 bootstrap
tests/         pytest checks
data/splits/   shared split and fold indices
configs/       frozen configuration
results/       predictions, tables, figures, experiment logs
models/        fitted pipelines
report/        final report
references/    BibTeX
docs/          data dictionary and data checks
```

## Contributions

Shi Xi: Logistic Regression and Gaussian Naive Bayes; data checks and preprocessing; shared split, evaluation protocol, cross-validation and freeze-then-test pipeline; calibration and uncertainty analyses; feature interpretation; report drafting and integration.
Li Erteng: KNN and Decision Tree; verification of the frozen configuration; experimental-design section; rank-comparison analysis; corresponding slides.
Jiang Chengrui: Random Forest and AdaBoost; threshold optimisation and cost-sensitivity analysis; out-of-fold cost figures; results section on thresholds; corresponding slides.
Overall approximately 40% / 30% / 30%.

## Data source

Yeh, I.-C. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H
