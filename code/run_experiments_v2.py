"""Dataset A (annual AAER benchmark of Bao et al., 2020): fits every model on chronological training
windows under three episode treatments and writes held-out scores.

Run:  python run_experiments_v2.py --data data_FraudDetection_JAR2020.csv --out results [--design annual|block] [--jobs N] [--rounds 1000]

Episode treatments (column "labels" in the output):
  asis    training labels as released
  recode  training labels of enforcement episodes (p_aaer) that continue into the test year are set to 0
  drop    those training rows are removed (episode-disjoint; primary specification)
Design: annual (train 1991..t-2, test t, t = 2003..2014) or block (four three-year test blocks).
Histogram gradient boosting is fitted with early stopping disabled so that no random validation split is drawn.
"""
import argparse, hashlib, json, platform
from pathlib import Path
import numpy as np, pandas as pd, sklearn
from joblib import Parallel, delayed
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

RAW = 'act ap at ceq che cogs csho dlc dltis dltt dp ib invt ivao ivst lct lt ni ppegt pstk re rect sale sstk txp txt xint prcc_f'.split()
RATIO = 'dch_wc ch_rsst dch_rec dch_inv soft_assets ch_cs ch_cm ch_roa issue bm dpi reoa EBIT ch_fcf'.split()
SEED = 42


def slog(a):
    return np.sign(a) * np.log1p(np.abs(a))


class RUSBoost:
    """Reimplementation of RUSBoost (random undersampling + discrete AdaBoost).
    Not the authors' MATLAB code: an independent reimplementation under the same inputs."""
    def __init__(self, n_rounds=3000, lr=0.1, min_leaf=5, max_leaf_nodes=None, seed=SEED):
        self.n_rounds, self.lr, self.min_leaf, self.max_leaf_nodes, self.seed = n_rounds, lr, min_leaf, max_leaf_nodes, seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32); y = np.asarray(y).astype(int)
        rng = np.random.default_rng(self.seed)
        pos = np.flatnonzero(y == 1); neg = np.flatnonzero(y == 0)
        s = 2 * y - 1
        w = np.full(len(y), 1.0 / len(y))
        self.trees, self.alphas = [], []
        for r in range(self.n_rounds):
            idx = np.concatenate([pos, rng.choice(neg, size=len(pos), replace=False)])
            t = DecisionTreeClassifier(min_samples_leaf=self.min_leaf, max_leaf_nodes=self.max_leaf_nodes,
                                       random_state=int(rng.integers(1 << 31)))
            t.fit(X[idx], y[idx], sample_weight=w[idx] / w[idx].sum())
            h = 2 * t.predict(X) - 1
            err = float(w[h != s].sum())
            if err <= 0 or err >= 0.5:          # degenerate learner: reset weights, keep going
                w = np.full(len(y), 1.0 / len(y)); continue
            a = self.lr * 0.5 * np.log((1 - err) / err)
            w = w * np.exp(-a * s * h); w /= w.sum()
            self.trees.append(t); self.alphas.append(a)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        f = np.zeros(len(X))
        for t, a in zip(self.trees, self.alphas):
            f += a * (2 * t.predict(X) - 1)
        f /= max(sum(self.alphas), 1e-12)
        return np.column_stack([(1 - f) / 2, (1 + f) / 2])


def make_models(n_rounds):
    return {
        'logit_raw':   (RAW,   lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))),
        'logit_slog':  (RAW,   lambda: make_pipeline(FunctionTransformer(slog), StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))),
        'logit_ratio': (RATIO, lambda: make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(max_iter=3000, C=1e6))),
        'hgb':         (RAW,   lambda: HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15, learning_rate=0.07,
                                                                      l2_regularization=10, early_stopping=False, random_state=SEED)),
        'rusboost':    (RAW,   lambda: RUSBoost(n_rounds=n_rounds, min_leaf=5, max_leaf_nodes=None)),
        'rusboost_s10':(RAW,   lambda: RUSBoost(n_rounds=n_rounds, min_leaf=5, max_leaf_nodes=11)),
    }


def one(df, train_end, test_years, labels, name, cols, factory):
    train = (df.fyear >= 1991) & (df.fyear <= train_end)
    test = df.fyear.isin(test_years)
    y = df.loc[train, 'misstate'].copy()
    n_recoded = 0
    Xtr = df.loc[train, cols]
    if labels in ('recode', 'drop'):
        span = set(df.loc[test & (df.misstate == 1), 'p_aaer'])
        hit = df.loc[train, 'p_aaer'].isin(span) & (y == 1)
        n_recoded = int(hit.sum())
        if labels == 'recode': y[hit] = 0
        else: y = y[~hit]; Xtr = Xtr[~hit]
    m = factory().fit(Xtr.to_numpy(), y.to_numpy())
    part = df.loc[test, ['gvkey', 'fyear', 'misstate', 'p_aaer']].copy()
    part['score'] = m.predict_proba(df.loc[test, cols].to_numpy())[:, 1]
    part['model'] = name; part['labels'] = labels; part['train_end'] = train_end
    part['unseen_firm'] = ~part.gvkey.isin(df.loc[train, 'gvkey'])
    part['train_pos_recoded'] = n_recoded
    return part


def run(data, out, design, jobs, n_rounds, label_sets=('asis', 'recode', 'drop'), tag=''):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data)
    assert set(RAW + RATIO + ['gvkey', 'fyear', 'misstate', 'p_aaer']).issubset(df.columns)
    assert set(df.misstate.unique()) <= {0, 1}
    assert not df.duplicated(['gvkey', 'fyear']).any()
    assert df[RAW].isna().sum().sum() == 0 and np.isfinite(df[RAW].to_numpy()).all()
    assert df.loc[df.misstate == 0, 'p_aaer'].isna().all() and df.loc[df.misstate == 1, 'p_aaer'].notna().all()
    if design == 'annual':
        windows = [(t - 2, [t]) for t in range(2003, 2015)]
    else:
        windows = [(2000, [2003, 2004, 2005]), (2003, [2006, 2007, 2008]), (2006, [2009, 2010, 2011]), (2009, [2012, 2013, 2014])]
    models = make_models(n_rounds)
    tasks = [(e, ty, lab, n, c, f) for e, ty in windows for lab in label_sets for n, (c, f) in models.items()]
    parts = Parallel(n_jobs=jobs, verbose=5)(delayed(one)(df, *t) for t in tasks)
    pd.concat(parts).to_csv(out / f'predictions_{design}{tag}.csv', index=False)
    audit = dict(rows=len(df), firms=int(df.gvkey.nunique()), positives=int(df.misstate.sum()), episodes=int(df.p_aaer.nunique()),
                 sha256=hashlib.sha256(Path(data).read_bytes()).hexdigest(), python=platform.python_version(),
                 sklearn=sklearn.__version__, pandas=pd.__version__, numpy=np.__version__, design=design,
                 windows=windows, rusboost_rounds=n_rounds, seed=SEED)
    (out / f'audit_{design}{tag}.json').write_text(json.dumps(audit, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--design', default='annual', choices=['annual', 'block'])
    ap.add_argument('--jobs', type=int, default=-1); ap.add_argument('--rounds', type=int, default=3000)
    ap.add_argument('--labels', default='asis,recode,drop'); ap.add_argument('--tag', default='')
    a = ap.parse_args(); run(a.data, a.out, a.design, a.jobs, a.rounds, tuple(a.labels.split(',')), a.tag)
