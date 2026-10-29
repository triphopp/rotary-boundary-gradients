"""Frozen iteration03 worker. One case per process; raw data retained."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import *  # noqa: E402,F401,F403  (repository layout)
setup()
import sys,json,gc,random,time,importlib,traceback,hashlib
from pathlib import Path
import torch,numpy as np,m3env,m3_emulator as E
from gpu_worker_1017 import to_torch
from manual_adjoint_reference import forward_adjoint
import local_rotation_triton as L
torch.set_num_threads(1)
shape_id,seed,slow=map(int,sys.argv[1:4]);shapes=[(1,1023,8,1,64,64),(1,2048,16,4,64,64),(2,4097,8,2,64,32),(1,8192,16,4,64,32)]
B,T,H,Hq,N,P=shapes[shape_id];tag=f'{shape_id}_{seed}_{slow}'
dest=RESULTS/'final_study'/f'{tag}.json';cv=lambda x:x.detach().double().cpu().numpy()
def load(name):
    m3env.use_source(KERNELS/name)
    return importlib.import_module('mamba_ssm.ops.triton.mamba3.mamba3_siso_combined')
low=load('terminal_fused');mid=load('midprecision_baseline');policy=load('boundary_policy');high=load('fp32_tuned');tf=load('tf32_baseline')
methods=[('bf16_sum',low,'none','bf16',64),('bf16_fp32_intermediates',mid,'none','bf16',64),('bf16_boundary_legacy',low,'legacy','bf16',64),('bf16_boundary_policy',policy,None,'bf16',64),('fp32_sum',high,'none','fp32',64),('fp32_storage_tf32_dots',tf,'none','fp32',32),('local_fp32',None,None,'bf16',8)]
inp=E.make_inputs(seed,B,T,H,Hq,N,P,'bf16',True,True,False)
if slow:inp['ADT']=(inp['ADT']*.01).astype(np.float32)
base=to_torch(inp,'bf16','cuda',torch)
cot=torch.from_numpy(np.random.default_rng(seed+91000).standard_normal((B,T,H,P))).cuda().bfloat16()
npi={k:cv(v) if v is not None else None for k,v in base.items()}
refcot=[cv(cot),np.zeros((B,H,N//4)),np.zeros((B,H,P,N)),np.zeros((B,H,N)),np.zeros((B,H,P))]
rf,ref=forward_adjoint(npi,refcot,checkpoint_size=64);del npi,refcot
result=dict(complete=False,environment=m3env.environment(),shape_id=shape_id,shape=shapes[shape_id],seed=seed,slow_decay=bool(slow),protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),methods={})
def save():dest.write_text(json.dumps(result,indent=2))
def metrics(gr,y=None):
    errors={}
    for k in ('Angles','DT'):
        got=cv(gr[k]);truth=ref[k];norm=np.linalg.norm(truth);rms=np.sqrt(np.mean(truth**2));mask=np.abs(truth)>.001*rms
        errors[k]=dict(rel_l2=float(np.linalg.norm(got-truth)/max(norm,1e-300)),max_abs=float(np.max(np.abs(got-truth))),reference_norm=float(norm),cosine=float(np.sum(got*truth)/max(np.linalg.norm(got)*norm,1e-300)),sign_disagreement=float(np.mean(np.sign(got[mask])!=np.sign(truth[mask]))),sign_mask_count=int(mask.sum()),finite=bool(np.isfinite(got).all()))
    if y is not None:errors['forward_rel_l2']=float(np.linalg.norm(cv(y)-rf[0])/np.linalg.norm(rf[0]))
    return errors
def elapsed(fn,n):
    events=[];wall=[]
    for _ in range(n):
        torch.cuda.synchronize();s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True);tic=time.perf_counter();s.record();value=fn();e.record();e.synchronize()
        wall.append((time.perf_counter()-tic)*1000);events.append(s.elapsed_time(e));del value
    return dict(event_ms=events,wall_ms=wall)
graphs={};keepers=[];order=list(methods);random.Random(511+shape_id).shuffle(order)
for name,module,mode,dtype,chunk in order:
    try:
        warm=torch.cuda.Stream();t=to_torch(inp,dtype,'cuda',torch,requires_grad=True)
        warm.wait_stream(torch.cuda.current_stream())
        def fwd(t=t,module=module,chunk=chunk):
            if module is None:return L.forward(t,chunk)
            return module.mamba3_siso_combined(**t,chunk_size=chunk,return_final_states=False).bfloat16(),None
        torch.cuda.synchronize();before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        with torch.cuda.stream(warm):y,states=fwd()
        torch.cuda.current_stream().wait_stream(warm)
        args=[v for v in t.values() if v is not None]
        def bwd(t=t,module=module,mode=mode,chunk=chunk,y=y,states=states,args=args):
            if module is None:return L.backward(t,states,cot,chunk)
            if mode is not None:module.CARRY_MODE=mode
            gs=torch.autograd.grad(y,args,cot,retain_graph=True,allow_unused=True)
            return {k:g for k,g in zip((k for k,v in t.items() if v is not None),gs)}
        with torch.cuda.stream(warm):gr=bwd()
        torch.cuda.synchronize();peak=(torch.cuda.max_memory_allocated()-before)/2**20
        errors=metrics(gr,y);initial={k:gr[k].clone() for k in ('Angles','DT')};del gr
        assert errors['Angles']['finite'] and errors['DT']['finite']
        ordinary=elapsed(bwd,9);forward=elapsed(fwd,5)
        promotion=None
        if dtype=='fp32':promotion=elapsed(lambda:[v.float() for v in base.values() if v is not None and v.dtype==torch.bfloat16],5)
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(3):bwd()
        torch.cuda.current_stream().wait_stream(warm);torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=warm):captured=bwd()
        graph.replay();torch.cuda.synchronize()
        replay_errors=metrics(captured)
        replay_equal={k:bool(torch.equal(initial[k],captured[k])) for k in ('Angles','DT')}
        assert all(replay_equal.values()),('capture mismatch',replay_equal)
        row=dict(errors=errors,replay_errors=replay_errors,replay_equal=replay_equal,extra_peak_MB=peak,ordinary_backward=ordinary,ordinary_forward=forward,promotion=promotion,chunk_or_vtile=chunk,timing_ms=[])
        result['methods'][name]=row;graphs[name]=graph;keepers.append((y,states,t,args,captured,bwd,fwd,warm));del initial
        print('READY',tag,name,errors['Angles']['rel_l2'],flush=True);save()
    except Exception:
        result['methods'][name]=dict(error=traceback.format_exc());save();raise
rng=random.Random(8271+seed+shape_id+slow)
for rep in range(9):
    names=list(graphs);rng.shuffle(names)
    for name in names:
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
        for _ in range(3):graphs[name].replay()
        end.record();end.synchronize();result['methods'][name]['timing_ms'].append(start.elapsed_time(end)/3)
for row in result['methods'].values():row['median_ms']=float(np.median(row['timing_ms']))
result['complete']=True;save()
print('COMPLETE',tag,{k:v['median_ms'] for k,v in result['methods'].items()},flush=True)
