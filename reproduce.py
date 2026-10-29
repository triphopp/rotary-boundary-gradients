"""Reproduction entry point. Run from anywhere: python reproduce.py --quick"""
import argparse, os, subprocess, sys
from pathlib import Path
from paths import ROOT, REFERENCE, EXPERIMENTS

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--quick', action='store_true', help='CPU: identity, adjoints, checkpoint replay (no GPU)')
ap.add_argument('--verify', action='store_true', help='CPU: recompute every paper number from raw results; independent checks')
ap.add_argument('--figures', action='store_true', help='regenerate summaries and figures from raw results')
ap.add_argument('--gpu-check', action='store_true', help='GPU: local-adjoint and policy validation')
ap.add_argument('--gpu-study', action='store_true', help='GPU: frozen 24-case x 7-method study, then analysis')
ap.add_argument('--resources', action='store_true', help='GPU: warmed memory / complete-primitive timing')
a = ap.parse_args()
if not any(vars(a).values()):
    ap.error('choose at least one mode')
if a.gpu_check or a.gpu_study or a.resources:
    import triton
    cc = Path(triton.__file__).parent / 'runtime' / 'tcc' / 'tcc.exe'
    if os.name == 'nt' and cc.exists():
        os.environ['CC'] = str(cc)
    os.environ.setdefault('TRITON_CACHE_DIR', str(ROOT / '.triton_cache'))
jobs = []
if a.quick:
    jobs += [REFERENCE / f for f in ('check_boundary_identity.py', 'manual_adjoint_reference.py',
                                     'validate_checkpoint_reference.py', 'local_rotation_adjoint.py')]
if a.verify:
    jobs += [ROOT / 'analysis' / 'recompute_from_logs.py', ROOT / 'verification' / 'theory_check.py']
if a.gpu_check:
    jobs += [EXPERIMENTS / 'validate_local_triton.py', EXPERIMENTS / 'validate_boundary_policy.py']
if a.gpu_study:
    jobs += [EXPERIMENTS / 'run_final_study.py']
if a.resources:
    jobs += [EXPERIMENTS / 'run_resource_e2e.py']
if a.figures or a.gpu_study or a.resources:
    jobs += [ROOT / 'analysis' / f for f in ('analyze_final_study.py', 'prepare_manuscript_results.py', 'make_scaling_figure.py')]
for job in jobs:
    args = [sys.executable, str(job)]
    if job.name == 'theory_check.py':
        args.append(str(ROOT / 'verification' / 'results' / 'theory_check.json'))
    print('RUN', job.relative_to(ROOT), flush=True)
    r = subprocess.run(args, cwd=job.parent)
    if r.returncode:
        raise SystemExit(f'FAILED {job.name} (exit {r.returncode})')
print('REPRODUCTION PASS')
