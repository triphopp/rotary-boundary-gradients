import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import *  # noqa: E402,F401,F403  (repository layout)
setup()
import sys,json,itertools
from pathlib import Path
import torch,numpy as np
from check_boundary_identity import inputs,sequential
import local_rotation_triton_fma as L
torch.set_num_threads(1)
records=[]
for T,da,vp in itertools.product((1,7,65),(2,3),(4,8,16)):
    args=inputs(T,6,da,False,76000+T)
    # Match the actual quantized kernel operands and output cotangent.
    low={'Q','K','V','Trap','Q_bias','K_bias','Z'}
    dev={k:v.detach().to('cuda',dtype=torch.bfloat16 if k in low else torch.float32).contiguous() if v is not None else None for k,v in args.items()}
    ref={k:v.detach().double().cpu().requires_grad_() if v is not None else None for k,v in dev.items()}
    out=sequential(**ref)[0];cot=torch.randn_like(out).bfloat16();(out*cot.double()).sum().backward()
    y,st=L.forward(dev,vp);g=L.backward(dev,st,cot.cuda(),vp)
    errors={};round_errors={}
    for k,v in ref.items():
        if v is None:continue
        truth=v.grad.numpy();got=g[k].double().cpu().numpy()
        errors[k]=float(np.linalg.norm(got-truth)/max(1e-12,np.linalg.norm(truth)))
        # BF16 stored gradients should differ from rounded exact gradients by at most a few ulps.
        rounded=v.grad.to(dev[k].dtype).double().numpy()
        round_errors[k]=float(np.max(np.abs(got-rounded))/max(1.,np.max(np.abs(truth))))
    ye=float(np.max(np.abs(y.double().cpu().numpy()-out.detach().numpy()))/max(1.,float(out.detach().abs().max())))
    record=dict(T=T,da=da,vp=vp,errors=errors,round_errors=round_errors,forward_error=ye)
    # Near-zero angle gradients use absolute checks, not ill-conditioned relative error.
    assert max(round_errors.values())<2e-4 and ye<.005,record
    records.append(record);print('PASS',T,da,vp,max(round_errors.values()),flush=True)
(VALIDATION/'local_triton_fma_validation.json').write_text(json.dumps(dict(status='PASS',cases=len(records),records=records),indent=2))
