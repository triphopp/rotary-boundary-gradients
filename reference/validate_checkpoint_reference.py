import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import *  # noqa: E402,F401,F403  (repository layout)
setup()
import json,numpy as np
from pathlib import Path
from manual_adjoint_reference import forward_adjoint
from check_boundary_identity import inputs,sequential
import torch
torch.set_num_threads(1);records=[]
for T in (1,7,65,257):
    for initial in (False,True):
        args=inputs(T,6,2,initial,9100+T);out=sequential(**args)
        cot=[torch.randn_like(x).detach().numpy() for x in out]
        cv=lambda x:x.detach().numpy()
        inp={k:(tuple(cv(x) for x in v) if isinstance(v,tuple) else cv(v) if v is not None else None) for k,v in args.items()}
        rf,rg=forward_adjoint(inp,cot)
        for size in (1,3,64,128):
            cf,cg=forward_adjoint(inp,cot,checkpoint_size=size)
            exact=all(np.array_equal(x,y) for x,y in zip(rf,cf)) and all(np.array_equal(rg[k],cg[k]) for k in rg)
            assert exact,(T,initial,size)
            records.append(dict(T=T,initial=initial,checkpoint=size,bitwise_equal=True))
result=dict(status='PASS',cases=len(records),records=records)
(VALIDATION/'checkpoint_reference_validation.json').write_text(json.dumps(result,indent=2))
print('PASS',len(records),'checkpoint comparisons, bitwise identical forward/adjoint.')
