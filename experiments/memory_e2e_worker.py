"""Fresh-process, warmed peak memory and complete forward/backward latency.
Uses primary seed1013 cases already in the frozen study; no new tuning.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import *  # noqa: E402,F401,F403  (repository layout)
setup()
import sys,json,importlib,time,gc
from pathlib import Path
import torch,numpy as np,m3env,m3_emulator as E
from gpu_worker_1017 import to_torch
import local_rotation_triton as L
torch.set_num_threads(1)
shape=int(sys.argv[1]);name=sys.argv[2];seed=1013
shapes=[(1,1023,8,1,64,64),(1,2048,16,4,64,64),(2,4097,8,2,64,32),(1,8192,16,4,64,32)]
B,T,H,Hq,N,P=shapes[shape]
spec={'bf16_sum':('terminal_fused','none',64),'bf16_fp32_intermediates':('midprecision_baseline','none',64),'bf16_boundary_legacy':('terminal_fused','legacy',64),'bf16_boundary_policy':('boundary_policy',None,64),'fp32_sum':('fp32_tuned','none',64),'fp32_storage_tf32_dots':('tf32_baseline','none',32),'local_fp32':(None,None,8)}
tree,mode,chunk=spec[name];module=None
if tree:
    m3env.use_source(KERNELS/tree);module=importlib.import_module('mamba_ssm.ops.triton.mamba3.mamba3_siso_combined')
    if mode is not None:module.CARRY_MODE=mode
base=to_torch(E.make_inputs(seed,B,T,H,Hq,N,P,'bf16',True,True,False),'bf16','cuda',torch,requires_grad=True)
cot=torch.from_numpy(np.random.default_rng(seed+91000).standard_normal((B,T,H,P))).cuda().bfloat16()
args=[v for v in base.values() if v is not None];keys=[k for k,v in base.items() if v is not None]
torch.cuda.synchronize();input_alloc=torch.cuda.memory_allocated();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
def step():
    if module is None:
        y,states=L.forward(base,chunk);gr=L.backward(base,states,cot,chunk);return y,gr
    # Promotion is inside the measured API and differentiable back to BF16 caller leaves.
    t={k:(v.float() if v is not None else None) for k,v in base.items()} if name.startswith('fp32_') else base
    y=module.mamba3_siso_combined(**t,chunk_size=chunk,return_final_states=False).bfloat16()
    gs=torch.autograd.grad(y,args,cot,allow_unused=True)
    return y,dict(zip(keys,gs))
with torch.cuda.stream(stream):
    for _ in range(5):
        tmp=step();del tmp
torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache()
peaks=[]
for _ in range(3):
    torch.cuda.reset_peak_memory_stats()
    with torch.cuda.stream(stream):tmp=step()
    torch.cuda.synchronize();peaks.append((torch.cuda.max_memory_allocated()-input_alloc)/2**20);del tmp
    gc.collect()
with torch.cuda.stream(stream):y,gr=step()
expected=json.loads((RESULTS/'final_study'/f'{shape}_{seed}_0.json').read_text())['methods'][name]
ref=np.load(RESULTS/'resource_reference'/f'{shape}.npz')
cv=lambda x:x.detach().double().cpu().numpy()
torch.cuda.synchronize();errors={k:float(np.linalg.norm(cv(gr[k])-ref[k])/np.linalg.norm(ref[k])) for k in ('Angles','DT')}
assert all(abs(errors[k]-expected['errors'][k]['rel_l2'])<=max(1e-8,expected['errors'][k]['rel_l2']*1e-4) for k in errors),(errors,expected['errors'])
del y,gr
events=[];wall=[]
for _ in range(15):
    torch.cuda.synchronize();s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True);tic=time.perf_counter()
    with torch.cuda.stream(stream):s.record();tmp=step();e.record()
    e.synchronize();wall.append((time.perf_counter()-tic)*1000);events.append(s.elapsed_time(e));del tmp
graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(graph,stream=stream):captured=step()
torch.cuda.synchronize();times=[]
for _ in range(15):
    s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True);s.record()
    for j in range(3):graph.replay()
    e.record();e.synchronize();times.append(s.elapsed_time(e)/3)
capture_errors={k:float(np.linalg.norm(cv(captured[1][k])-ref[k])/np.linalg.norm(ref[k])) for k in ('Angles','DT')}
assert errors==capture_errors,(errors,capture_errors)
result=dict(shape=shape,seed=seed,method=name,environment=m3env.environment(),complete=True,errors=errors,capture_errors=capture_errors,input_alloc_MB=input_alloc/2**20,warmed_extra_peak_MB=peaks,ordinary_e2e_event_ms=events,ordinary_e2e_wall_ms=wall,captured_e2e_ms=times,median_captured_e2e_ms=float(np.median(times)))
(RESULTS/'resource_e2e'/f'{shape}_{name}.json').write_text(json.dumps(result,indent=2));print('PASS',shape,name,np.median(peaks),np.median(times),flush=True)
