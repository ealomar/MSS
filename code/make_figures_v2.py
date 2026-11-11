"""Figures for the manuscript. Run: python make_figures_v2.py --res results --res2 results_ds2 --out fig"""
import argparse
from pathlib import Path
import pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

M = {'rusboost': ('RUSBoost', '#2a78d6', 'o', '-'), 'hgb': ('Gradient boosting', '#eb6834', 's', '--'),
     'logit_slog': ('Signed-log logit', '#1baf7a', '^', '-.'), 'logit_ratio': ('Ratio logit', '#eda100', 'D', ':'),
     'logit_raw': ('Raw logit', '#e87ba4', 'v', (0, (5, 1)))}
INK, MUTE = '#1a1a19', '#6b6a63'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': MUTE, 'axes.labelcolor': INK, 'xtick.color': MUTE,
                     'ytick.color': MUTE, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e5df',
                     'grid.linewidth': 0.6, 'legend.frameon': False, 'axes.axisbelow': True, 'savefig.dpi': 400, 'savefig.bbox': 'tight'})


def fig_design(out):
    fig, ax = plt.subplots(figsize=(7.2, 2.5)); ax.axis('off'); ax.set_xlim(0, 100); ax.set_ylim(0, 40)
    top = ['Company-year\nrecords', 'Chronological\ntraining window', 'Episode\ntreatment', 'Model\nfitting']
    bot = ['Test-year\nscoring', 'Annual review\ncapacity', 'Capture\nmeasurement', 'Firm-cluster\nbootstrap']
    w, h = 19, 11
    def box(x, y, t):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1.2', fc='#f3f2ee', ec=MUTE, lw=0.9))
        ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=8.5, color=INK)
    xs = [2, 27, 52, 77]
    for x, t in zip(xs, top): box(x, 26, t)
    for x, t in zip(xs, bot): box(x, 3, t)
    ar = dict(arrowstyle='-|>', color=MUTE, lw=1.0, shrinkA=0, shrinkB=0)
    for a, b in zip(xs[:-1], xs[1:]):
        ax.annotate('', xy=(b - 0.6, 31.5), xytext=(a + w + 0.6, 31.5), arrowprops=ar)
        ax.annotate('', xy=(b - 0.6, 8.5), xytext=(a + w + 0.6, 8.5), arrowprops=ar)
    xm, x0 = xs[3] + w / 2, xs[0] + w / 2
    ax.plot([xm, xm, x0], [25.4, 20, 20], color=MUTE, lw=1.0, solid_capstyle='butt')
    ax.annotate('', xy=(x0, 14.6), xytext=(x0, 20.05), arrowprops=ar)
    fig.savefig(out / 'fig1_design.png'); plt.close(fig)


def fig_capture(res, out):
    c = pd.read_csv(res / 'capacity_2003_2008.csv')
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for ax, lab, ttl in zip(axs, ['asis', 'drop'], ['(a)', '(b)']):
        for m, (nm, col, mk, ls) in M.items():
            d = c[(c.labels == lab) & (c.model == m)]
            ax.plot(d.capacity * 100, d.detected, color=col, marker=mk, ms=3.5, lw=1.6, ls=ls, label=nm, markevery=2)
        d = c[(c.labels == lab) & (c.model == 'hgb')]
        ax.plot(d.capacity * 100, d.random_expected, color=MUTE, lw=1.0, label='Random review')
        ax.set_xlabel('Annual review capacity (%)'); ax.set_title(ttl, loc='left', fontsize=9, color=INK)
    axs[0].set_ylabel('Misstatement company-years captured')
    axs[1].legend(fontsize=7.5, loc='upper left')
    fig.tight_layout(); fig.savefig(out / 'fig2_capture.png'); plt.close(fig)


def fig_carry(res, out):
    l = pd.read_csv(res / 'leakage_drop_2003_2008.csv')
    fig, ax = plt.subplots(figsize=(7.2, 3.0)); caps = [0.01, 0.03, 0.05, 0.10]; bw = 0.15
    for i, (m, (nm, col, mk, ls)) in enumerate(M.items()):
        d = l[l.model == m].set_index('capacity').loc[caps]
        x = [j + (i - 2) * (bw + 0.02) for j in range(4)]
        ax.bar(x, d.inflation, width=bw, color=col, label=nm, edgecolor='white', linewidth=0.8)
        ax.errorbar(x, d.inflation, yerr=[d.inflation - d.lo, d.hi - d.inflation], fmt='none', ecolor=INK, elinewidth=0.8, capsize=2)
    ax.axhline(0, color=MUTE, lw=0.8); ax.set_xticks(range(4)); ax.set_xticklabels(['1%', '3%', '5%', '10%'])
    ax.set_xlabel('Annual review capacity'); ax.set_ylabel('Episode carry-over (company-years)')
    ax.grid(axis='x', visible=False); ax.legend(fontsize=7.5, ncol=3, loc='upper left')
    fig.tight_layout(); fig.savefig(out / 'fig3_carryover.png'); plt.close(fig)


MB = {'rusboost': ('RUSBoost', '#2a78d6', 'o', '-'), 'hgb': ('Gradient boosting', '#eb6834', 's', '--'),
      'logit_amount': ('Amount logit', '#1baf7a', '^', '-.'), 'logit_ratio': ('Ratio logit', '#eda100', 'D', ':'),
      'hgb_ratio': ('Gradient boosting (ratios)', '#e87ba4', 'v', (0, (5, 1)))}


def fig_capture_b(res2, out):
    c = pd.read_csv(res2 / 'capacity_all.csv')
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for ax, lab, ttl in zip(axs, ['asis', 'drop'], ['(a)', '(b)']):
        for m, (nm, col, mk, ls) in MB.items():
            d = c[(c['labels'] == lab) & (c.model == m)]
            ax.plot(d.capacity * 100, d.detected, color=col, marker=mk, ms=3.5, lw=1.6, ls=ls, label=nm, markevery=2)
        d = c[(c['labels'] == lab) & (c.model == 'hgb')]
        ax.plot(d.capacity * 100, d.random_expected, color=MUTE, lw=1.0, label='Random review')
        ax.set_xlabel('Annual review capacity (%)'); ax.set_title(ttl, loc='left', fontsize=9, color=INK)
    axs[0].set_ylabel('Misstatement firm-quarters captured')
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, fontsize=7.5, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(out / 'fig5_capture_b.png'); plt.close(fig)


def fig_carry_b(res2, out):
    l = pd.read_csv(res2 / 'leakage_drop_all.csv')
    fig, ax = plt.subplots(figsize=(7.2, 3.0)); caps = [0.01, 0.03, 0.05, 0.10]; bw = 0.15
    for i, (m, (nm, col, mk, ls)) in enumerate(MB.items()):
        d = l[l.model == m].set_index('capacity').loc[caps]
        x = [j + (i - 2) * (bw + 0.02) for j in range(4)]
        ax.bar(x, d.inflation, width=bw, color=col, label=nm, edgecolor='white', linewidth=0.8)
        ax.errorbar(x, d.inflation, yerr=[d.inflation - d.lo, d.hi - d.inflation], fmt='none', ecolor=INK, elinewidth=0.8, capsize=2)
    ax.axhline(0, color=MUTE, lw=0.8); ax.set_xticks(range(4)); ax.set_xticklabels(['1%', '3%', '5%', '10%'])
    ax.set_xlabel('Annual review capacity'); ax.set_ylabel('Episode carry-over (firm-quarters)')
    ax.grid(axis='x', visible=False); ax.legend(fontsize=7.5, ncol=3, loc='upper left')
    fig.tight_layout(); fig.savefig(out / 'fig6_carryover_b.png'); plt.close(fig)


def fig_paired(res, out):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, m, ttl in zip(axs, ['rusboost', 'hgb'], ['(a)', '(b)']):
        t = pd.read_csv(res / f'paired_drop_{m}_vs_logit_ratio_2003_2008.csv'); col = M[m][1]
        ax.fill_between(t.capacity * 100, t.lo, t.hi, color=col, alpha=0.18, lw=0)
        ax.plot(t.capacity * 100, t['diff'], color=col, lw=1.8, marker=M[m][2], ms=3.5)
        ax.axhline(0, color=MUTE, lw=0.8); ax.set_xlabel('Annual review capacity (%)'); ax.set_title(ttl, loc='left', fontsize=9, color=INK)
    axs[0].set_ylabel('Capture difference vs. ratio logit')
    fig.tight_layout(); fig.savefig(out / 'fig4_paired.png'); plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--res', required=True); ap.add_argument('--out', required=True); ap.add_argument('--res2', default=None)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); res = Path(a.res)
    fig_design(out); fig_capture(res, out); fig_carry(res, out); fig_paired(res, out)
    if a.res2: fig_capture_b(Path(a.res2), out); fig_carry_b(Path(a.res2), out)
