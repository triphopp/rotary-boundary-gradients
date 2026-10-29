import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import *  # noqa: E402,F401,F403  (repository layout)
setup()
import sys,subprocess,json,random
from pathlib import Path
D=RESULTS/'final_study';D.mkdir(exist_ok=True)
jobs=[(shape,seed,slow) for slow in (0,1) for shape in range(4) for seed in (1009,1013,1019)]
random.Random(67173).shuffle(jobs)
statuses=[]
for shape,seed,slow in jobs:
    tag=f'{shape}_{seed}_{slow}';print('START',tag,flush=True)
    with (D/f'{tag}.log').open('w') as log:
        try:
            p=subprocess.run([sys.executable,str(EXPERIMENTS/'benchmark_final_worker.py'),str(shape),str(seed),str(slow)],stdout=log,stderr=subprocess.STDOUT,timeout=900)
            status=dict(shape=shape,seed=seed,slow=slow,returncode=p.returncode)
        except subprocess.TimeoutExpired:status=dict(shape=shape,seed=seed,slow=slow,timeout=True)
    statuses.append(status);(D/'run_status.json').write_text(json.dumps(statuses,indent=2));print('DONE',status,flush=True)
    # Stop on harness/correctness/resource failure; do not contaminate subsequent testing.
    if status.get('returncode',1)!=0:raise RuntimeError(status)
