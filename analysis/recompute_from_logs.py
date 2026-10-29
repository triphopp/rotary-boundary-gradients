"""Independent recomputation of every quantitative claim in the manuscript from the raw logs.
Usage:  python analysis/recompute_from_logs.py
Reads only raw JSON/CSV (final_study/*.json, resource_e2e/*.json, terminal_heldout_metrics.csv,
boundary_identity_results.json); does not import the authors' analysis code.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import *  # noqa: E402,F401,F403  (repository layout)
setup()

import json, glob, csv, sys, os, itertools, hashlib, collections
import numpy as np
M = ['bf16_sum', 'bf16_fp32_intermediates', 'bf16_boundary_legacy', 'bf16_boundary_policy',
     'fp32_storage_tf32_dots', 'fp32_sum', 'local_fp32']
NONB = ['bf16_sum', 'bf16_fp32_intermediates', 'fp32_sum', 'fp32_storage_tf32_dots', 'local_fp32']
cases = [json.load(open(f)) for f in sorted(glob.glob(str(RESULTS / 'final_study' / '*_*_*.json')))]
sha = hashlib.sha256(open(PROTOCOL, 'rb').read()).hexdigest()
print('cases', len(cases), 'all complete', all(c['complete'] for c in cases),
      'protocol hash matches', all(c['protocol_sha256'] == sha for c in cases),
      'all replay_equal', all(all(m['replay_equal'].values()) for c in cases for m in c['methods'].values()))
ang = lambda c, m: c['methods'][m]['errors']['Angles']['rel_l2']
dt = lambda c, m: c['methods'][m]['errors']['DT']['rel_l2']
tm = lambda c, m: c['methods'][m]['median_ms']
ok = lambda c, m, a=0.025: ang(c, m) <= a and dt(c, m) <= 0.01

def exact_cluster_boot(d):
    meds = [np.median(sum((d[g] for g in combo), [])) for combo in itertools.product(sorted(d), repeat=len(d))]
    return np.quantile(meds, [.025, .975]), len(set(np.round(meds, 9)))

for slow in (False, True):
    cs = [c for c in cases if c['slow_decay'] == slow]
    print('\n==== %s (%d cases)' % ('slow-decay stress' if slow else 'primary', len(cs)))
    print('%-26s %9s %9s %12s %9s' % ('method', 'medAngle', 'maxAngle', 'pass1/2.5/5', 'medTime'))
    for m in M:
        print('%-26s %9.4g %9.4g %12s %9.4g' % (m, np.median([ang(c, m) for c in cs]), max(ang(c, m) for c in cs),
              '%d/%d/%d' % tuple(sum(ok(c, m, a) for c in cs) for a in (0.01, 0.025, 0.05)),
              np.median([tm(c, m) / tm(c, 'bf16_sum') for c in cs])))
    red = {s: [min(ang(c, 'bf16_sum'), ang(c, 'bf16_fp32_intermediates')) / ang(c, 'bf16_boundary_policy')
               for c in cs if c['shape_id'] == s] for s in range(4)}
    allred = sum(red.values(), [])
    q, nd = exact_cluster_boot(red)
    print('error reduction: median %.4f, min %.3f, max %.3f, worst regression %.3f; exact 4-cluster bootstrap %.2f-%.2f (%d distinct medians)'
          % (np.median(allred), min(allred), max(allred), max(1 / x for x in allred), q[0], q[1], nd))
    sp = collections.defaultdict(list); fastest = []
    for c in cs:
        if not ok(c, 'bf16_boundary_policy'): continue
        el = [m for m in NONB if ok(c, m)]
        f = min(el, key=lambda m: tm(c, m)); fastest.append(f); sp[c['shape_id']].append(tm(c, f) / tm(c, 'bf16_boundary_policy'))
    allsp = sum(sp.values(), [])
    q, nd = exact_cluster_boot(dict(sp))
    print('qualified speedup: median %.4f (n=%d) range %.3f-%.3f; fastest qualifying = %s; exact bootstrap %.2f-%.2f'
          % (np.median(allsp), len(allsp), min(allsp), max(allsp), dict(collections.Counter(fastest)), q[0], q[1]))
    ov = [tm(c, 'bf16_boundary_policy') / tm(c, 'bf16_sum') for c in cs]
    print('policy/BF16 captured overhead: median %.4f max %.4f min %.4f' % (np.median(ov), max(ov), min(ov)))
    ev = lambda c, m: np.median(c['methods'][m]['ordinary_backward']['event_ms'])
    print('ordinary (eager, event-timed) backward: TF32 faster than policy in %d/%d cases; policy/BF16 eager range %.2f-%.2f'
          % (sum(ev(c, 'fp32_storage_tf32_dots') < ev(c, 'bf16_boundary_policy') for c in cs), len(cs),
             min(ev(c, 'bf16_boundary_policy') / ev(c, 'bf16_sum') for c in cs), max(ev(c, 'bf16_boundary_policy') / ev(c, 'bf16_sum') for c in cs)))
    T = np.array([c['shape'][1] for c in cs], float)
    for m in ('bf16_sum', 'bf16_fp32_intermediates', 'bf16_boundary_legacy', 'bf16_boundary_policy', 'fp32_storage_tf32_dots'):
        e = np.array([ang(c, m) for c in cs]); b, a = np.polyfit(np.log(T), np.log(e), 1)
        print('  scaling %-24s log-log slope vs T = %+.3f' % (m, b))
    for s in range(4):
        sub = [c for c in cs if c['shape_id'] == s]; Tn = sub[0]['shape'][1]
        print('  shape %d T=%d last chunk=%d: legacy/policy=%.2f, reduction=%.2f, reduction/sqrt(T/1023)=%.2f, timing max/min over seeds: policy %.2f TF32 %.2f'
              % (s, Tn, Tn % 64 or 64, np.median([ang(c, 'bf16_boundary_legacy') / ang(c, 'bf16_boundary_policy') for c in sub]),
                 np.median(red[s]) if not slow else np.median([min(ang(c, 'bf16_sum'), ang(c, 'bf16_fp32_intermediates')) / ang(c, 'bf16_boundary_policy') for c in sub]),
                 (np.median(red[s]) if not slow else 0) / np.sqrt(Tn / 1023),
                 max(tm(c, 'bf16_boundary_policy') for c in sub) / min(tm(c, 'bf16_boundary_policy') for c in sub),
                 max(tm(c, 'fp32_storage_tf32_dots') for c in sub) / min(tm(c, 'fp32_storage_tf32_dots') for c in sub)))
    if not slow:
        e = np.array([ang(c, 'fp32_storage_tf32_dots') for c in cs]); b, a = np.polyfit(np.log(T), np.log(e), 1)
        pol = np.median([ang(c, 'bf16_boundary_policy') for c in cs])
        print('  extrapolation: TF32 error reaches policy median at T~%.0f and 2.5%% at T~%.0f' % ((pol / np.exp(a)) ** (1 / b), (0.025 / np.exp(a)) ** (1 / b)))
r = {(s, m): json.load(open(RESULTS / 'resource_e2e' / f'{s}_{m}.json')) for s in range(4) for m in M}
print('\n==== resource_e2e')
for s in range(4):
    pk = lambda m: float(np.median(r[s, m]['warmed_extra_peak_MB'])); c = lambda m: r[s, m]['median_captured_e2e_ms']
    print('shape %d: policy peak %.3f MiB, policy/BF16 peak %.5f, captured TF32/policy %.3f, captured policy/BF16 %.3f'
          % (s, pk('bf16_boundary_policy'), pk('bf16_boundary_policy') / pk('bf16_sum'), c('fp32_storage_tf32_dots') / c('bf16_boundary_policy'), c('bf16_boundary_policy') / c('bf16_sum')))
rows = list(csv.DictReader(open(RESULTS / 'terminal_study' / 'terminal_heldout_metrics.csv')))
by = collections.defaultdict(dict)
for x in rows: by[(x['T'], x['initial'], x['terminal_objective'], x['seed'])][x['mode']] = float(x['angle_rel_l2'])
longr = [v['legacy'] / v['terminal'] for k, v in by.items() if int(k[0]) >= 2049]
print('\n==== iteration02 terminal study: %d cases, terminal worse in %d, >20%% worse in %d (max ratio %.3f), long median legacy/terminal %.6f, lengths %s'
      % (len(by), sum(v['terminal'] > v['legacy'] for v in by.values()), sum(v['terminal'] > 1.2 * v['legacy'] for v in by.values()),
         max(v['terminal'] / v['legacy'] for v in by.values()), np.median(longr), sorted({int(k[0]) for k in by})))
bi = json.load(open(VALIDATION / 'boundary_identity_results.json'))['summary']
print('==== identity validation JSON:', {k: bi[k] for k in ('cases', 'boundary_checks', 'max_identity_error', 'max_gradient_error', 'max_fd_error', 'max_error_if_terminal_omitted')})
