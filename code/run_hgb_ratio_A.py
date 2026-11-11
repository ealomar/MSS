"""Dataset A: gradient boosting on the 14 ratios, as released and episode-disjoint, test years 2003-2008.
Run:  python run_hgb_ratio_A.py --data data_FraudDetection_JAR2020.csv --out results"""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from run_experiments_v2 import RATIO, SEED

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); df = pd.read_csv(a.data); parts = []
    for t in range(2003, 2009):
        tr = (df.fyear >= 1991) & (df.fyear <= t - 2); te = df.fyear == t
        span = set(df.loc[te & (df.misstate == 1), 'p_aaer'])
        for lab in ['asis', 'drop']:
            keep = tr if lab == 'asis' else tr & ~(df.p_aaer.isin(span) & (df.misstate == 1))
            m = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15, learning_rate=0.07, l2_regularization=10,
                                               early_stopping=False, random_state=SEED).fit(df.loc[keep, RATIO], df.loc[keep, 'misstate'])
            g = df.loc[te, ['gvkey', 'fyear', 'misstate', 'p_aaer']].copy(); g['score'] = m.predict_proba(df.loc[te, RATIO])[:, 1]
            g['model'] = 'hgb_ratio'; g['labels'] = lab; parts.append(g)
    pd.concat(parts).to_csv(Path(a.out) / 'predictions_hgb_ratio.csv', index=False)
