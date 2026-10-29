import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import *  # noqa: E402,F401,F403  (repository layout)
setup()
import sys,subprocess,json,random
from pathlib import Path
import torch,numpy as np,m3_emulator as E
from gpu_worker_1017 import to_torch
from manual_adjoint_reference import forward_adjoint
torch.set_num_threads(1);(RESULTS/'resource_e2e').mkdir(exist_ok=True);(RESULTS/'resource_reference').mkdir(exist_ok=True)
cv=lambda x:x.detach().double().cpu().numpy()
for shape,(B,T,H,Hq,N,P) in enumerate([(1,1023,8,1,64,64),(1,2048,16,4,64,64),(2,4097,8,2,64,32),(1,8192,16,4,64,32)]):
    t=to_torch(E.make_inputs(1013,B,T,H,Hq,N,P,'bf16',True,True,False),'bf16','cpu',torch)
    npi={k:cv(v) if v is not None else None for k,v in t.items()}
    cot=torch.from_numpy(np.random.default_rng(1013+91000).standard_normal((B,T,H,P))).bfloat16()
    rf,gr=forward_adjoint(npi,[cv(cot),np.zeros((B,H,N//4)),np.zeros((B,H,P,N)),np.zeros((B,H,N)),np.zeros((B,H,P))],checkpoint_size=64)
    np.savez(RESULTS/'resource_reference'/f'{shape}.npz',Angles=gr['Angles'],DT=gr['DT']);del rf,gr,t,npi,cot
methods=['bf16_sum','bf16_fp32_intermediates','bf16_boundary_legacy','bf16_boundary_policy','fp32_sum','fp32_storage_tf32_dots','local_fp32']
jobs=[(s,m) for s in range(4) for m in methods];random.Random(39011).shuffle(jobs);statuses=[]
for shape,name in jobs:
    print('START',shape,name,flush=True)
    with (RESULTS/'resource_e2e'/f'{shape}_{name}.log').open('w') as log:
        p=subprocess.run([sys.executable,str(EXPERIMENTS/'memory_e2e_worker.py'),str(shape),name],stdout=log,stderr=subprocess.STDOUT,timeout=600)
    statuses.append(dict(shape=shape,method=name,returncode=p.returncode));(RESULTS/'resource_e2e'/'run_status.json').write_text(json.dumps(statuses,indent=2));print('DONE',statuses[-1],flush=True)
    if p.returncode:raise RuntimeError(statuses[-1])
