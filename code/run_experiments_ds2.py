"""Second dataset: SEC XBRL quarterly financials with AAER labels (Waffo Dzuyo et al., Hugging Face
WaguyMZ/Financial_statements_fraud_dataset). Same episode-disjoint, capacity-resolved protocol as
run_experiments_v2.py, on firm-quarters.
Run:  python run_experiments_ds2.py --root C:\\fraud_dataset --out results_ds2 [--roles all|respondent]
"""
import argparse, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd, sklearn
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from run_experiments_v2 import RUSBoost, slog, SEED

TEST_YEARS = list(range(2014, 2021))


class Clip(BaseEstimator, TransformerMixin):
    """Winsorise each column at the training 1st and 99th percentiles."""
    def fit(self, X, y=None):
        self.lo_, self.hi_ = np.nanpercentile(X, 1, axis=0), np.nanpercentile(X, 99, axis=0); return self
    def transform(self, X): return np.clip(X, self.lo_, self.hi_)


def load(root, roles):
    root = Path(root)
    f = pd.read_csv(root / 'FINANCIALS_PREPROCESSED/sec_financials_quarterly.csv', low_memory=False)
    a = pd.read_csv(root / 'AAER_PREPROCESSED/df_aaer_with_misstatements_since_2009.csv', low_memory=False)
    a['quarter'] = a.fiscal_quarter.str[-2:]
    if roles == 'respondent': a = a[a.role.isin(['respondent', 'defendant'])]
    ep = a.groupby(['cik', 'year', 'quarter']).aaer_no.agg(lambda s: '|'.join(sorted(set(s)))).rename('releases').reset_index()
    f = f.merge(ep, on=['cik', 'year', 'quarter'], how='left')
    f['misstate'] = f.releases.notna().astype(int)
    f['p_aaer'] = f.releases.str.split('|').str[0]
    f = f.sort_values(['cik', 'year', 'quarter']).reset_index(drop=True)
    ratio = [c for c in f.columns if c.startswith(('ratio_', 'Beneish_'))]
    amount = [c for c in f.columns if c.startswith(('agg_', 'diff_'))]
    X = f[ratio + amount].replace([np.inf, -np.inf], np.nan)
    f[ratio + amount] = X.fillna(0.0)
    return f, ratio, amount


def make_models(ratio, amount, rounds):
    hgb = lambda: HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15, learning_rate=0.07, l2_regularization=10,
                                                 early_stopping=False, random_state=SEED)
    return {
        'logit_ratio':  (ratio,  lambda: make_pipeline(Clip(), StandardScaler(), LogisticRegression(max_iter=5000, C=1e6))),
        'logit_amount': (amount, lambda: make_pipeline(FunctionTransformer(slog), StandardScaler(), LogisticRegression(max_iter=5000, C=0.1))),
        'hgb_ratio':    (ratio,  hgb),
        'hgb_amount':   (amount, hgb),
        'hgb':          (ratio + amount, hgb),
        'rusboost':     (ratio + amount, lambda: RUSBoost(n_rounds=rounds, min_leaf=5)),
    }


def one(df, t, labels, name, cols, factory):
    train = (df.year <= t - 2).to_numpy(); test = (df.year == t).to_numpy()
    y = df.loc[train, 'misstate'].copy(); Xtr = df.loc[train, cols]; n_hit = 0
    if labels in ('recode', 'drop'):
        span = set('|'.join(df.loc[test & (df.misstate == 1), 'releases']).split('|')) - {''}
        rel = df.loc[train, 'releases'].fillna('')
        hit = rel.map(lambda s: bool(s) and not span.isdisjoint(s.split('|')))
        n_hit = int(hit.sum())
        if labels == 'recode': y[hit] = 0
        else: y = y[~hit]; Xtr = Xtr[~hit]
    m = factory().fit(Xtr.to_numpy(), y.to_numpy())
    part = df.loc[test, ['cik', 'year', 'quarter', 'misstate', 'p_aaer']].rename(columns={'cik': 'gvkey', 'year': 'fyear'}).copy()
    part['score'] = m.predict_proba(df.loc[test, cols].to_numpy())[:, 1]
    part['model'] = name; part['labels'] = labels; part['train_pos_removed'] = n_hit; part['train_pos'] = int(y.sum())
    return part


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--roles', default='all', choices=['all', 'respondent']); ap.add_argument('--rounds', type=int, default=1000)
    ap.add_argument('--jobs', type=int, default=-1)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df, ratio, amount = load(a.root, a.roles); models = make_models(ratio, amount, a.rounds)
    tasks = [(t, lab, n, c, f) for t in TEST_YEARS for lab in ('asis', 'drop', 'recode') for n, (c, f) in models.items()]
    parts = Parallel(n_jobs=a.jobs, verbose=5)(delayed(one)(df, *x) for x in tasks)
    pd.concat(parts).to_csv(out / f'predictions_{a.roles}.csv', index=False)
    fin = Path(a.root) / 'FINANCIALS_PREPROCESSED/sec_financials_quarterly.csv'
    (out / f'audit_{a.roles}.json').write_text(json.dumps(dict(rows=len(df), firms=int(df.cik.nunique()), positives=int(df.misstate.sum()),
        fraud_firms=int(df[df.misstate == 1].cik.nunique()), releases=int(df.p_aaer.nunique()), n_ratio=len(ratio), n_amount=len(amount),
        test_years=TEST_YEARS, roles=a.roles, rounds=a.rounds, sklearn=sklearn.__version__,
        sha256_financials=hashlib.sha256(fin.read_bytes()).hexdigest()), indent=2))
