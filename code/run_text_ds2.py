"""Second dataset, text modality: TF-IDF + logit on summarised MD&A sections, on the firm-quarters that have a
summary, under as-released and episode-disjoint training. A gradient-boosting model on the tabular features of
the same firm-quarters is fitted for reference.
Run:  python run_text_ds2.py --root C:\\fraud_dataset --mda DIR_WITH_UNZIPPED_quarterly --out results_ds2
"""
import argparse, os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from run_experiments_ds2 import load, TEST_YEARS, SEED

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True); ap.add_argument('--mda', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df, ratio, amount = load(a.root, 'all')
    files = {tuple(x[:-4].split('_')): x for x in os.listdir(a.mda) if x.endswith('.txt')}
    key = list(zip(df.cik.astype(str), df.year.astype(str) + df.quarter))
    df['file'] = [files.get(k) for k in key]
    df = df[df.file.notna()].reset_index(drop=True)
    df['text'] = [Path(a.mda, f).read_text(encoding='utf8', errors='ignore') for f in df.file]
    parts = []
    for t in TEST_YEARS:
        test = (df.year == t).to_numpy()
        for lab in ('asis', 'drop'):
            train = (df.year <= t - 2).to_numpy().copy()
            if lab == 'drop':
                span = set('|'.join(df.loc[test & (df.misstate == 1), 'releases']).split('|')) - {''}
                hit = df.releases.fillna('').map(lambda s: bool(s) and not span.isdisjoint(s.split('|'))).to_numpy()
                train &= ~hit
            y = df.loc[train, 'misstate'].to_numpy()
            models = {
                'text_logit': (make_pipeline(TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.9, ngram_range=(1, 2), max_features=200000),
                                             LogisticRegression(C=10.0, max_iter=3000)), 'text'),
                'hgb_tabular': (HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15, learning_rate=0.07, l2_regularization=10,
                                                               early_stopping=False, random_state=SEED), ratio + amount)}
            for name, (m, cols) in models.items():
                Xtr = df.loc[train, cols]; Xte = df.loc[test, cols]
                m.fit(Xtr if cols == 'text' else Xtr.to_numpy(), y)
                p = df.loc[test, ['cik', 'year', 'quarter', 'misstate', 'p_aaer']].rename(columns={'cik': 'gvkey', 'year': 'fyear'}).copy()
                p['score'] = m.predict_proba(Xte if cols == 'text' else Xte.to_numpy())[:, 1]; p['model'] = name; p['labels'] = lab
                parts.append(p)
        print(t, int(test.sum()), int(df.loc[test, 'misstate'].sum()), flush=True)
    pd.concat(parts).to_csv(out / 'predictions_text.csv', index=False)
