"""Fig. 1 of the arXiv version: angle-gradient error vs sequence length (log-log), from raw final_study JSON.
Usage: python analysis/make_scaling_figure.py [final_study_dir] [out_prefix]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import *  # noqa: E402,F401,F403  (repository layout)
setup()

import json, glob, sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else str(RESULTS / 'final_study')
out = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / 'paper' / 'images' / 'error_vs_length')
cases = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(src, '*_*_*.json')))]
# categorical slots 1-5 (validated light palette); policy gets slot 1
SER = [('bf16_boundary_policy', 'Boundary policy', '#2a78d6', 'D'),
       ('bf16_sum', 'BF16 suffix', '#eb6834', 'o'),
       ('bf16_fp32_intermediates', 'FP32 intermediates', '#1baf7a', 's'),
       ('bf16_boundary_legacy', 'Legacy boundary', '#eda100', '^'),
       ('fp32_storage_tf32_dots', 'FP32 / TF32', '#e87ba4', 'v')]
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e',
                     'ytick.color': '#52514e', 'savefig.dpi': 300})
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), sharey=True)
fits = {}
for ax, slow in zip(axes, (False, True)):
    cs = [c for c in cases if c['slow_decay'] == slow]
    Ts = sorted({c['shape'][1] for c in cs})
    for key, label, color, mk in SER:
        T = np.array([c['shape'][1] for c in cs], float)
        e = np.array([c['methods'][key]['errors']['Angles']['rel_l2'] for c in cs])
        slope, icpt = np.polyfit(np.log(T), np.log(e), 1)
        fits[(slow, key)] = slope
        med = [np.median([c['methods'][key]['errors']['Angles']['rel_l2'] for c in cs if c['shape'][1] == t]) for t in Ts]
        ax.scatter(T, e, s=9, color=color, alpha=0.35, linewidths=0, zorder=2)
        lw = 2.0 if key == 'bf16_boundary_policy' else 1.2
        ax.plot(Ts, med, color=color, lw=lw, marker=mk, ms=4.5, mec='#fcfcfb', mew=0.8, zorder=3,
                label=label)
    ax.axhline(0.025, color='#52514e', ls=(0, (4, 3)), lw=0.8, zorder=1)
    ax.text(Ts[0], 0.0262, '2.5% target', color='#52514e', fontsize=7, va='bottom')
    ax.set_xscale('log', base=2); ax.set_yscale('log')
    ax.set_xticks(Ts, [str(t) for t in Ts]); ax.minorticks_off()
    ax.set_xlabel('Sequence length T (shape also varies)')
    ax.set_title('Slow-decay stress (ADT x 0.01)' if slow else 'Primary cases', fontsize=8.5, color='#0b0b0b')
    ax.grid(alpha=0.18, which='major', lw=0.5)
axes[0].set_ylabel('Angle-gradient relative L2 error')
axes[0].set_ylim(1.2e-3, 0.45)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=5, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.01), handlelength=2.2)
fig.tight_layout(w_pad=1.2, rect=(0, 0.08, 1, 1))
for ext in ('pdf', 'png'):
    fig.savefig(f'{out}.{ext}', bbox_inches='tight', facecolor='#ffffff')
print(json.dumps({('slow' if k[0] else 'primary') + ':' + k[1]: round(v, 3) for k, v in fits.items()}, indent=0))
