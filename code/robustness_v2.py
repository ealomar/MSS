"""RUSBoost robustness on the 2003-2008 test years: extra seeds at 1000 rounds and one run at 3000 rounds.
Run:  python robustness_v2.py --data PATH --out DIR [--jobs N]"""
import argparse, math
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from run_experiments_v2 import RAW, RUSBoost


def fit(df, t, labels, seed, rounds):
    train = (df.fyear >= 1991) & (df.fyear <= t - 2); test = df.fyear == t
    y = df.loc[train, 'misstate'].copy(); X = df.loc[train, RAW]
    if labels == 'drop':
        hit = df.loc[train, 'p_aaer'].isin(set(df.loc[test & (df.misstate == 1), 'p_aaer'])) & (y == 1)
        y = y[~hit]; X = X[~hit]
    m = RUSBoost(n_rounds=rounds, seed=seed).fit(X.to_numpy(), y.to_numpy())
    g = df.loc[test, ['gvkey', 'fyear', 'misstate', 'p_aaer']].copy(); g['score'] = m.predict_proba(df.loc[test, RAW].to_numpy())[:, 1]
    rows = []
    for q in [0.01, 0.03, 0.05, 0.10]:
        c = g.sort_values(['score', 'gvkey'], ascending=[False, True]).head(math.ceil(q * len(g)))
        rows.append(dict(year=t, labels=labels, seed=seed, rounds=rounds, capacity=q, detected=int(c.misstate.sum())))
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True); ap.add_argument('--jobs', type=int, default=-1)
    a = ap.parse_args(); df = pd.read_csv(a.data)
    cfg = [(s, 1000) for s in (1, 2, 3)] + [(42, 3000)]
    tasks = [(t, lab, s, r) for s, r in cfg for t in range(2003, 2009) for lab in ('asis', 'drop')]
    out = Parallel(n_jobs=a.jobs, verbose=5)(delayed(fit)(df, *x) for x in tasks)
    r = pd.DataFrame([z for o in out for z in o])
    r.to_csv(Path(a.out) / 'robustness_rusboost_by_year.csv', index=False)
    r.groupby(['rounds', 'seed', 'labels', 'capacity']).detected.sum().unstack('capacity').to_csv(Path(a.out) / 'robustness_rusboost.csv')
