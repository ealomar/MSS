"""Analysis of held-out scores: benchmark metrics, capacity curves, paired firm-cluster bootstrap,
crossover capacity, selection regret, episode-level capture.
Run:  python analyze_v2.py --res DIR [--boot 2000]
"""
import argparse, json, math
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

GRID = [round(x, 3) for x in np.arange(0.005, 0.1001, 0.005)]
MODELS = ['logit_raw', 'logit_slog', 'logit_ratio', 'hgb', 'rusboost', 'rusboost_s10']


def topk(g, q):
    k = math.ceil(q * len(g))
    return g.sort_values(['score', 'gvkey'], ascending=[False, True]).head(k)


def ndcg_at_k(g, k):
    s = g.sort_values(['score', 'gvkey'], ascending=[False, True]).misstate.to_numpy()[:k]
    disc = 1 / np.log2(np.arange(2, k + 2))
    ideal = disc[:min(k, int(g.misstate.sum()))].sum()
    return float((s * disc).sum() / ideal) if ideal > 0 else np.nan


def bench(p, years):
    rows = []
    for (lab, m), d in p[p.fyear.isin(years)].groupby(['labels', 'model']):
        per = []
        for y, g in d.groupby('fyear'):
            k = math.ceil(0.01 * len(g)); c = topk(g, 0.01)
            per.append(dict(auc=roc_auc_score(g.misstate, g.score), ndcg=ndcg_at_k(g, k),
                            sens=c.misstate.sum() / g.misstate.sum(), prec=c.misstate.mean()))
        per = pd.DataFrame(per)
        rows.append(dict(labels=lab, model=m, AUC=per.auc.mean(), NDCG_at_k=per.ndcg.mean(),
                         sensitivity_top1=per.sens.mean(), precision_top1=per.prec.mean(),
                         pooled_AUC=roc_auc_score(d.misstate, d.score)))
    return pd.DataFrame(rows)


def capacity(p, years):
    rows = []
    for (lab, m), d in p[p.fyear.isin(years)].groupby(['labels', 'model']):
        tot_pos = int(d.misstate.sum()); tot_ep = d.loc[d.misstate == 1, 'p_aaer'].nunique()
        for q in GRID:
            ch = pd.concat([topk(g, q) for _, g in d.groupby('fyear')])
            rows.append(dict(labels=lab, model=m, capacity=q, reviewed=len(ch), positives=tot_pos,
                             detected=int(ch.misstate.sum()), episodes=tot_ep,
                             episodes_detected=int(ch.loc[ch.misstate == 1, 'p_aaer'].nunique()),
                             random_expected=sum(g.misstate.sum() * math.ceil(q * len(g)) / len(g) for _, g in d.groupby('fyear'))))
    r = pd.DataFrame(rows)
    r['precision'] = r.detected / r.reviewed; r['recall'] = r.detected / r.positives
    r['episode_recall'] = r.episodes_detected / r.episodes
    return r


class Boot:
    """Firm-cluster bootstrap with integer firm multiplicities as row weights.
    Weighted top-k: rows are taken in score order until the weighted count reaches ceil(q * sum w)."""
    def __init__(self, p, years, models):
        d = p[p.fyear.isin(years)]
        base = d[d.model == models[0]].sort_values(['fyear', 'gvkey']).reset_index(drop=True)
        self.firms, self.fidx = np.unique(base.gvkey, return_inverse=True)
        self.y = base.misstate.to_numpy(float); self.year = base.fyear.to_numpy()
        self.years = sorted(set(years)); self.models = models
        self.score = {}
        for m in models:
            s = d[d.model == m].sort_values(['fyear', 'gvkey']).reset_index(drop=True)
            assert (s.gvkey.to_numpy() == base.gvkey.to_numpy()).all()
            self.score[m] = s.score.to_numpy()
        self.rows = {t: np.flatnonzero(self.year == t) for t in self.years}
        self.order = {(m, t): self.rows[t][np.argsort(-self.score[m][self.rows[t]], kind='stable')] for m in models for t in self.years}

    def capture(self, w, m, t, qs):
        o = self.order[(m, t)]; wo = w[o]; cw = np.cumsum(wo); cy = np.cumsum(wo * self.y[o])
        out = []
        for q in qs:
            k = math.ceil(q * cw[-1]); j = int(np.searchsorted(cw, k))
            j = min(j, len(o) - 1)
            prev_w = cw[j - 1] if j > 0 else 0.0; prev_y = cy[j - 1] if j > 0 else 0.0
            out.append(prev_y + (k - prev_w) * self.y[o[j]] if wo[j] > 0 else prev_y)
        return np.array(out)

    def weights(self, rng):
        c = np.bincount(rng.integers(0, len(self.firms), len(self.firms)), minlength=len(self.firms))
        return c[self.fidx].astype(float)

    def auc(self, w, m, t):
        r = self.rows[t]; ww = w[r]; keep = ww > 0
        y = self.y[r][keep]
        if y.min() == y.max(): return np.nan
        return roc_auc_score(y, self.score[m][r][keep], sample_weight=ww[keep])


def crossover(diff):
    """Smallest grid capacity from which the difference stays positive up to 10%; nan if none."""
    pos = diff > 0
    if not pos[-1]: return np.nan
    i = len(pos) - 1
    while i > 0 and pos[i - 1]: i -= 1
    return GRID[i]


def paired(bt, a, b, B, rng):
    """Bootstrap of pooled capture difference a - b along the capacity grid."""
    def curve(w): return sum(bt.capture(w, a, t, GRID) - bt.capture(w, b, t, GRID) for t in bt.years)
    w1 = np.ones(len(bt.y)); pt = curve(w1)
    reps = np.array([curve(bt.weights(rng)) for _ in range(B)])
    qs = np.array([crossover(r) for r in reps])
    tab = pd.DataFrame(dict(capacity=GRID, diff=pt, lo=np.percentile(reps, 2.5, axis=0), hi=np.percentile(reps, 97.5, axis=0),
                            p_gt0=(reps > 0).mean(axis=0), p_lt0=(reps < 0).mean(axis=0)))
    i1, i10 = GRID.index(0.01), GRID.index(0.1)
    info = dict(a=a, b=b, crossover_point=crossover(pt), crossover_median=float(np.nanmedian(qs)) if np.isfinite(qs).any() else None,
                crossover_ci=[float(x) for x in np.nanpercentile(qs, [2.5, 97.5])] if np.isfinite(qs).any() else None,
                share_reps_with_crossover=float(np.isfinite(qs).mean()),
                share_reps_sign_flip_1_vs_10=float(((reps[:, i1] < 0) & (reps[:, i10] > 0)).mean()))
    return tab, info


def leakage(p, years, B, rng, clean='drop'):
    """Serial-episode inflation: pooled capture with training labels as released minus capture after
    removing test-spanning episodes from training, per model and capacity, with firm-cluster bootstrap CI.
    Also the difference in inflation between each model and the ratio logit."""
    d = p[p.labels.isin(['asis', clean])].copy(); d['model'] = d.model + '|' + d.labels
    names = [f'{m}|{l}' for m in MODELS for l in ['asis', clean]]
    bt = Boot(d, years, names); qs = [0.01, 0.03, 0.05, 0.10]
    def infl(w): return {m: sum(bt.capture(w, f'{m}|asis', t, qs) - bt.capture(w, f'{m}|{clean}', t, qs) for t in bt.years) for m in MODELS}
    pt = infl(np.ones(len(bt.y))); reps = [infl(bt.weights(rng)) for _ in range(B)]
    rows = []
    for m in MODELS:
        r = np.array([x[m] for x in reps]); rd = np.array([x[m] - x['logit_ratio'] for x in reps])
        for i, q in enumerate(qs):
            rows.append(dict(model=m, capacity=q, inflation=pt[m][i], lo=np.percentile(r[:, i], 2.5), hi=np.percentile(r[:, i], 97.5),
                             p_gt0=(r[:, i] > 0).mean(), excess_vs_ratio_logit=pt[m][i] - pt['logit_ratio'][i],
                             ex_lo=np.percentile(rd[:, i], 2.5), ex_hi=np.percentile(rd[:, i], 97.5), ex_p_gt0=(rd[:, i] > 0).mean()))
    return pd.DataFrame(rows)


def selection(bt, w, q, models):
    """Each year t (from the 2nd test year) pick a model from all earlier test years, by pooled-year mean AUC
    or by pooled capture at capacity q; apply it at t. Returns captures (auc_rule, cap_rule, per-model fixed)."""
    cap = {(m, t): bt.capture(w, m, t, [q])[0] for m in models for t in bt.years}
    auc = {(m, t): bt.auc(w, m, t) for m in models for t in bt.years}
    ra = rc = 0.0; picks = []
    for i, t in enumerate(bt.years[1:], 1):
        past = bt.years[:i]
        a_best = max(models, key=lambda m: np.nanmean([auc[(m, s)] for s in past]))
        c_best = max(models, key=lambda m: (sum(cap[(m, s)] for s in past), np.nanmean([auc[(m, s)] for s in past])))
        ra += cap[(a_best, t)]; rc += cap[(c_best, t)]; picks.append((t, a_best, c_best))
    fixed = {m: sum(cap[(m, t)] for t in bt.years[1:]) for m in models}
    return ra, rc, fixed, picks


def main(res, B, only_leak=False):
    res = Path(res); rng = np.random.default_rng(42); out = {}
    p = pd.concat([pd.read_csv(f) for f in sorted(res.glob('predictions_annual*.csv'))], ignore_index=True)
    P1, P2, ALL = list(range(2003, 2009)), list(range(2009, 2015)), list(range(2003, 2015))
    bench(p, P1).to_csv(res / 'benchmark_2003_2008.csv', index=False)
    for nm, yrs in [('2003_2008', P1), ('2009_2014', P2), ('2003_2014', ALL)]:
        capacity(p, yrs).to_csv(res / f'capacity_{nm}.csv', index=False)
    if 'drop' in set(p.labels):
        for nm, yrs in [('2003_2008', P1), ('2003_2014', ALL)]:
            for clean in ['drop', 'recode']:
                leakage(p, yrs, B, np.random.default_rng(7), clean).to_csv(res / f'leakage_{clean}_{nm}.csv', index=False)
        pdrop = p[p.labels == 'drop']; dsum = {}
        for nm, yrs in [('2003_2008', P1), ('2003_2014', ALL)]:
            btd = Boot(pdrop, yrs, MODELS)
            for a, b in [('rusboost', 'logit_ratio'), ('hgb', 'logit_ratio'), ('logit_slog', 'logit_ratio'), ('rusboost', 'hgb'), ('rusboost', 'logit_slog'), ('hgb', 'logit_slog')]:
                tab, info = paired(btd, a, b, B, np.random.default_rng(11))
                tab.to_csv(res / f'paired_drop_{a}_vs_{b}_{nm}.csv', index=False); dsum[f'{a}_vs_{b}_{nm}'] = info
        (res / 'paired_drop_summary.json').write_text(json.dumps(dsum, indent=2))
    if only_leak: return
    pr = p[p.labels == 'recode']
    for nm, yrs in [('2003_2008', P1), ('2003_2014', ALL)]:
        bt = Boot(pr, yrs, MODELS)
        for a, b in [('hgb', 'logit_raw'), ('hgb', 'logit_slog'), ('rusboost', 'hgb'), ('rusboost', 'logit_raw'), ('hgb', 'logit_ratio'), ('rusboost', 'logit_ratio')]:
            tab, info = paired(bt, a, b, B, rng)
            tab.to_csv(res / f'paired_{a}_vs_{b}_{nm}.csv', index=False); out[f'paired_{a}_vs_{b}_{nm}'] = info
        sel = {}
        for q in [0.01, 0.03, 0.05, 0.10]:
            ra, rc, fixed, picks = selection(bt, np.ones(len(bt.y)), q, MODELS)
            d = []
            for _ in range(max(B // 4, 200)):
                x = selection(bt, bt.weights(rng), q, MODELS); d.append(x[1] - x[0])
            sel[str(q)] = dict(auc_rule=ra, capacity_rule=rc, diff=rc - ra, diff_ci=[float(v) for v in np.percentile(d, [2.5, 97.5])],
                               p_cap_rule_better=float((np.array(d) > 0).mean()), fixed=fixed, best_fixed=max(fixed, key=fixed.get),
                               picks=[(int(t), a, c) for t, a, c in picks])
        out[f'selection_{nm}'] = sel
    (res / 'analysis_summary.json').write_text(json.dumps(out, indent=2))
    # Block design: capture under three-year test blocks
    pb = res / 'predictions_block.csv'
    if pb.exists():
        b = pd.read_csv(pb); rows = []
        for (lab, m), d in b.groupby(['labels', 'model']):
            for q in [0.01, 0.03, 0.05, 0.10]:
                ch = pd.concat([topk(g, q) for _, g in d.groupby('fyear')])
                rows.append(dict(labels=lab, model=m, capacity=q, reviewed=len(ch), detected=int(ch.misstate.sum()),
                                 episodes_detected=int(ch.loc[ch.misstate == 1, 'p_aaer'].nunique())))
        pd.DataFrame(rows).to_csv(res / 'capacity_block.csv', index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--res', required=True); ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--only-leak', action='store_true')
    a = ap.parse_args(); main(a.res, a.boot, a.only_leak)
