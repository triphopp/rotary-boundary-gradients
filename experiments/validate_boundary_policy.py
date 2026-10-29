import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import *  # noqa: E402,F401,F403  (repository layout)
setup()
import sys,json,itertools
from pathlib import Path
import torch,numpy as np,m3env,m3_emulator as E
from gpu_worker_1017 import to_torch
from manual_adjoint_reference import forward_adjoint
m3env.use_source(KERNELS/'boundary_policy')
import mamba_ssm.ops.triton.mamba3.mamba3_siso_combined as M
torch.set_num_threads(1);records=[]
conv=lambda x:x.detach().double().cpu().numpy()
for T,initial,term in itertools.product((1,63,64,65,2049),(False,True),(False,True)):
    inp=E.make_inputs(311,1,T,2,1,64,32,'bf16',True,True,initial)
    t=to_torch(inp,'bf16','cuda',torch,requires_grad=True)
    raw=M.mamba3_siso_combined(**t,chunk_size=64,return_final_states=term)
    out=list(raw) if term else [raw];rng=np.random.default_rng(91311)
    cot=[torch.from_numpy(rng.standard_normal(tuple(x.shape))).to(x.device).to(x.dtype) for x in out]
    rc=[conv(x) for x in cot]
    if not term:rc += [np.zeros((1,2,16)),np.zeros((1,2,32,64)),np.zeros((1,2,64)),np.zeros((1,2,32))]
    npi={k:(tuple(conv(x) for x in v) if isinstance(v,tuple) else conv(v) if v is not None else None) for k,v in t.items()}
    _,rg=forward_adjoint(npi,rc);torch.autograd.backward(out,cot)
    metrics={}
    for k in ('Angles','DT'):
        delta=conv(t[k].grad)-rg[k];rn=float(np.linalg.norm(rg[k]));dn=float(np.linalg.norm(delta))
        metrics[k]=dict(rel_l2=dn/max(rn,1e-300),error_l2=dn,ref_l2=rn,max_abs=float(np.max(np.abs(delta))))
        assert np.all(np.isfinite(delta))
        # Development regression guard, not the held-out superiority criterion.
        # Near-zero mathematical gradients must use an absolute criterion.
        assert (metrics[k]['max_abs']<1e-6 if rn<1e-10 else metrics[k]['rel_l2']<(.05 if k=='Angles' else .02)),(T,initial,term,k,metrics[k])
    row=dict(T=T,initial=initial,terminal=term,metrics=metrics);records.append(row);print(row,flush=True)
    (VALIDATION/'boundary_policy_validation.json').write_text(json.dumps(dict(complete=False,cases=records),indent=2))
(VALIDATION/'boundary_policy_validation.json').write_text(json.dumps(dict(complete=True,cases=records),indent=2))
print('COMPLETE',len(records),'policy cases; inspect short nearly-zero-gradient cases using absolute metrics.')
