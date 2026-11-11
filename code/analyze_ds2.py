"""Analysis for the second dataset; reuses the functions of analyze_v2.py.
Run:  python analyze_ds2.py --res results_ds2 [--boot 1000] [--tag all|respondent|text]"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import analyze_v2 as A

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--res', required=True); ap.add_argument('--boot', type=int, default=1000); ap.add_argument('--tag', default='all')
    a = ap.parse_args(); res = Path(a.res); tag = a.tag
    p = pd.read_csv(res / f'predictions_{tag}.csv')
    A.MODELS = ['text_logit', 'hgb_tabular'] if tag == 'text' else ['logit_ratio', 'logit_amount', 'hgb_ratio', 'hgb_amount', 'hgb', 'rusboost']
    years = sorted(p.fyear.unique())
    A.bench(p, years).to_csv(res / f'benchmark_{tag}.csv', index=False)
    A.capacity(p, years).to_csv(res / f'capacity_{tag}.csv', index=False)
    ref = 'hgb_tabular' if tag == 'text' else 'logit_ratio'
    # leakage() reports excess against 'logit_ratio'; alias the reference for the text run
    q = p.copy()
    if tag == 'text': q['model'] = q.model.replace({'hgb_tabular': 'logit_ratio'}); A.MODELS = ['text_logit', 'logit_ratio']
    for clean in (['drop'] if tag == 'text' else ['drop', 'recode']):
        A.leakage(q, years, a.boot, np.random.default_rng(7), clean).to_csv(res / f'leakage_{clean}_{tag}.csv', index=False)
    if tag != 'text':
        bt = A.Boot(p[p['labels'] == 'drop'], years, A.MODELS); out = {}
        for m in ['rusboost', 'hgb', 'hgb_ratio', 'hgb_amount', 'logit_amount']:
            tab, info = A.paired(bt, m, 'logit_ratio', a.boot, np.random.default_rng(11))
            tab.to_csv(res / f'paired_drop_{m}_vs_logit_ratio_{tag}.csv', index=False); out[m] = info
        (res / f'paired_drop_summary_{tag}.json').write_text(json.dumps(out, indent=2))
