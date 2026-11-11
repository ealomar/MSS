"""Model class versus input type: episode carry-over for linear and nonlinear models on the same inputs,
and the nonlinear-minus-linear excess, with firm-cluster bootstrap intervals, on both datasets.
Run:  python mechanism_v2.py --res results --res2 results_ds2 [--boot 1000]"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import analyze_v2 as A

QS = [0.01, 0.03, 0.05, 0.10]


def run(p, years, cells, B, seed):
    d = p[p['labels'].isin(['asis', 'drop'])].copy(); d['model'] = d.model + '|' + d['labels']
    models = sorted({m for pair in cells.values() for m in pair})
    bt = A.Boot(d, years, [f'{m}|{l}' for m in models for l in ('asis', 'drop')]); rng = np.random.default_rng(seed)
    def infl(w): return {m: sum(bt.capture(w, f'{m}|asis', t, QS) - bt.capture(w, f'{m}|drop', t, QS) for t in bt.years) for m in models}
    pt = infl(np.ones(len(bt.y))); reps = [infl(bt.weights(rng)) for _ in range(B)]
    rows = []
    for inp, (lin, non) in cells.items():
        for kind, f in [('linear', lambda x: x[lin]), ('nonlinear', lambda x: x[non]), ('nonlinear_minus_linear', lambda x: x[non] - x[lin])]:
            r = np.array([f(x) for x in reps]); v = f(pt)
            for i, q in enumerate(QS):
                rows.append(dict(input=inp, kind=kind, capacity=q, value=v[i], lo=np.percentile(r[:, i], 2.5), hi=np.percentile(r[:, i], 97.5), p_gt0=(r[:, i] > 0).mean()))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--res', required=True); ap.add_argument('--res2', required=True); ap.add_argument('--boot', type=int, default=1000)
    a = ap.parse_args(); r1, r2 = Path(a.res), Path(a.res2)
    p = pd.concat([pd.read_csv(f) for f in sorted(r1.glob('predictions_annual*.csv'))] + [pd.read_csv(r1 / 'predictions_hgb_ratio.csv')], ignore_index=True)
    p = p[p.fyear.between(2003, 2008)]
    run(p, list(range(2003, 2009)), {'amounts': ('logit_slog', 'hgb'), 'ratios': ('logit_ratio', 'hgb_ratio')}, a.boot, 7).assign(dataset='A').to_csv(r1 / 'mechanism_A.csv', index=False)
    q = pd.read_csv(r2 / 'predictions_all.csv')
    run(q, sorted(q.fyear.unique()), {'amounts': ('logit_amount', 'hgb_amount'), 'ratios': ('logit_ratio', 'hgb_ratio')}, a.boot, 7).assign(dataset='B').to_csv(r2 / 'mechanism_B.csv', index=False)
