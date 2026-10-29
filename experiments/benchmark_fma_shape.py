import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import *  # noqa: E402,F401,F403  (repository layout)
setup()
import sys,json,gc,random,time
from pathlib import Path
import torch,numpy as np,m3env,m3_emulator as E
from gpu_worker_1017 import to_torch
from manual_adjoint_reference import forward_adjoint
import local_rotation_triton_fma as L
torch.set_num_threads(1);cv=lambda x:x.detach().double().cpu().numpy()
sid=int(sys.argv[1]);B,H,Hq,N,P,T=[(1,8,1,64,64,1023),(1,16,4,64,64,2048),(2,8,2,64,32,4097),(1,16,4,64,32,8192)][sid]
inp=E.make_inputs(1013,B,T,H,Hq,N,P,'bf16',True,True,False)
t=to_torch(inp,'bf16','cuda',torch)
cot=torch.from_numpy(np.random.default_rng(92013).standard_normal((B,T,H,P))).cuda().bfloat16()
npi={k:cv(v) if v is not None else None for k,v in t.items()}
refcot=[cv(cot),np.zeros((B,H,N//4)),np.zeros((B,H,P,N)),np.zeros((B,H,N)),np.zeros((B,H,P))]
rf,ref=forward_adjoint(npi,refcot,checkpoint_size=64)
records={};graphs={};keep=[]
for vp in (8,):
    torch.cuda.synchronize();before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
    y,st=L.forward(t,vp);gr=L.backward(t,st,cot,vp);torch.cuda.synchronize()
    peak=torch.cuda.max_memory_allocated()-before
    errors={k:dict(rel_l2=float(np.linalg.norm(cv(gr[k])-ref[k])/np.linalg.norm(ref[k])),max_abs=float(np.max(np.abs(cv(gr[k])-ref[k])))) for k in ('Angles','DT')}
    errors['forward_rel_l2']=float(np.linalg.norm(cv(y)-rf[0])/np.linalg.norm(rf[0]))
    assert all(np.isfinite(errors[k]['rel_l2']) for k in ('Angles','DT'))
    del gr
    warm=torch.cuda.Stream();warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        for _ in range(3):L.backward(t,st,cot,vp)
    torch.cuda.current_stream().wait_stream(warm);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=warm):L.backward(t,st,cot,vp)
    torch.cuda.synchronize();graphs[vp]=graph;keep.append((y,st))
    ordinary=[];forward=[];wall=[]
    for _ in range(9):
        s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize();tic=time.perf_counter();s.record();L.backward(t,st,cot,vp);e.record();e.synchronize()
        wall.append((time.perf_counter()-tic)*1000);ordinary.append(s.elapsed_time(e))
        s.record();yy,ss=L.forward(t,vp);e.record();e.synchronize();forward.append(s.elapsed_time(e));del yy,ss
    records[vp]=dict(errors=errors,extra_peak_MB=peak/2**20,state_MB=st.numel()*st.element_size()/2**20,ordinary_event_ms=ordinary,ordinary_wall_ms=wall,forward_ms=forward,timing_ms=[])
    print('READY',vp,errors,flush=True)
rng=random.Random(648107)
for _ in range(15):
    order=list(graphs);rng.shuffle(order)
    for vp in order:
        s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True);s.record()
        for j in range(3):graphs[vp].replay()
        e.record();e.synchronize();records[vp]['timing_ms'].append(s.elapsed_time(e)/3)
for r in records.values():r['median_ms']=float(np.median(r['timing_ms']))
(RESULTS/'fma_shape_audit'/f'{sid}.json').write_text(json.dumps(dict(environment=m3env.environment(),shape=[B,T,H,Hq,N,P],seed=1013,records=records,complete=True),indent=2))
print('COMPLETE',{vp:r['median_ms'] for vp,r in records.items()},flush=True)
