# Episode-disjoint, capacity-resolved evaluation of misstatement screening models

Code and result tables for the paper *Detecting New Accounting Misstatement Episodes: An Episode-Disjoint
Evaluation of Machine Learning Screening* by Emadaldeen Hassan Alomar.

The code measures **episode carry-over**: the misstatement observations a screening model captures only because
earlier periods of the same enforcement episode were labelled in its training data. It refits each model with
those episodes removed from training (episode-disjoint evaluation), reviews the top 0.5-10% of each test year,
and reports capture in observations and in distinct episodes with firm-cluster bootstrap intervals.

## Repository layout

| Path | Content |
|---|---|
| `code/` | all scripts (run them from inside `code/`) |
| `results/dataset_A/` | result tables for the annual dataset |
| `results/dataset_B/` | result tables for the quarterly dataset, including the text subsample |
| `figures/` | the six figures of the paper |

Raw data and held-out score files (`predictions_*.csv`, about 300 MB in total) are not included; the scripts
regenerate the scores.

## Data (not redistributed)

**Dataset A** - Bao, Ke, Li, Yu and Zhang (2020), 146,045 U.S. company-years, 1990-2014.
Download `data_FraudDetection_JAR2020.csv` from https://github.com/JarFraud/FraudDetection
SHA-256: `ed4cefd9db2a23d0a13911d9f482fc755d3dcf6c9be0bc051af95ddcae5b286f`

**Dataset B** - Financial Statements Fraud Dataset, 269,097 firm-quarters from SEC XBRL filings, 2009-2025.
https://huggingface.co/datasets/WaguyMZ/Financial_statements_fraud_dataset

    pip install -U huggingface_hub
    python -c "from huggingface_hub import snapshot_download; snapshot_download('WaguyMZ/Financial_statements_fraud_dataset', repo_type='dataset', local_dir='fraud_dataset', max_workers=1)"

Then unzip `fraud_dataset/SEC_MDA_SUMMARIZED/quarterly.zip` (needed only for the text model).
SHA-256 of `FINANCIALS_PREPROCESSED/sec_financials_quarterly.csv`:
`bebb09cef9b4ce9c728353dc63bb179b99f372d38dd70f3b995c7cb03f93f781`

## Environment

Python 3.13, with the packages in `requirements.txt`:

    pip install -r requirements.txt

## Reproducing the results

Run from inside `code/`. `A` is the folder for Dataset A outputs and `B` for Dataset B outputs.

    cd code

    # Dataset A
    python run_experiments_v2.py --data ../data/data_FraudDetection_JAR2020.csv --out A --design annual --rounds 1000
    python run_experiments_v2.py --data ../data/data_FraudDetection_JAR2020.csv --out A --design block --rounds 1000 --labels asis,recode
    python run_hgb_ratio_A.py   --data ../data/data_FraudDetection_JAR2020.csv --out A
    python robustness_v2.py     --data ../data/data_FraudDetection_JAR2020.csv --out A
    python analyze_v2.py --res A --boot 1000

    # Dataset B
    python run_experiments_ds2.py --root ../fraud_dataset --out B --roles all
    python run_experiments_ds2.py --root ../fraud_dataset --out B --roles respondent
    python run_text_ds2.py --root ../fraud_dataset --mda ../fraud_dataset/SEC_MDA_SUMMARIZED/quarterly --out B
    python analyze_ds2.py --res B --tag all
    python analyze_ds2.py --res B --tag respondent
    python analyze_ds2.py --res B --tag text

    # Both datasets
    python mechanism_v2.py --res A --res2 B
    python make_figures_v2.py --res A --res2 B --out ../figures

Total runtime is about two hours on two CPU cores. All random seeds are fixed. RUSBoost is the only stochastic
model; `robustness_v2.py` repeats it with three further seeds and with 3000 boosting rounds.

## Scripts

| Script | Purpose |
|---|---|
| `run_experiments_v2.py` | Dataset A: six models, three episode treatments, annual or block design |
| `run_hgb_ratio_A.py` | Dataset A: gradient boosting on the 14 ratios |
| `robustness_v2.py` | Dataset A: RUSBoost seeds and boosting rounds |
| `analyze_v2.py` | benchmark metrics, capture by capacity, carry-over, paired differences, selection rules |
| `run_experiments_ds2.py` | Dataset B: six models crossing model class and input type |
| `run_text_ds2.py` | Dataset B: TF-IDF logit on MD&A summaries and tabular reference model |
| `analyze_ds2.py` | Dataset B analysis (reuses `analyze_v2.py`) |
| `mechanism_v2.py` | carry-over by model class and input type on both datasets |
| `make_figures_v2.py` | figures |

## Main result files

| File | Content |
|---|---|
| `capacity_*.csv` | observations and distinct episodes captured at each review capacity, per model and treatment |
| `leakage_drop_*.csv`, `leakage_recode_*.csv` | episode carry-over with 95% firm-cluster bootstrap intervals |
| `paired_drop_*_vs_logit_ratio_*.csv` | capture difference from the ratio logit under episode-disjoint evaluation |
| `benchmark_*.csv` | AUC, NDCG@k, top-1% sensitivity and precision |
| `mechanism_A.csv`, `mechanism_B.csv` | carry-over for linear and nonlinear models on the same inputs |
| `robustness_rusboost.csv` | RUSBoost capture by seed and number of rounds |

In the files, `labels` takes the values `asis` (as released), `drop` (episode-disjoint) and `recode`.

## Notes

RUSBoost is an independent reimplementation (random undersampling with discrete AdaBoost), not the original
MATLAB code of Bao et al.

## Author and contact

Emadaldeen Hassan Alomar, Department of Accounting and Finance, College of Business, Jazan University, Jazan, Saudi Arabia.
Contact: ealomar@jazanu.edu.sa

## License

Code: MIT (see `LICENSE`). The datasets keep their own licences and are not part of this repository.
