"""Established rotating-coordinate recurrence with a direct, local reverse adjoint.
Independent strong baseline; no originality claim. Output-only, zero initial state.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import *  # noqa: E402,F401,F403  (repository layout)
setup()
import numpy as np

def rowrot(x,angle):
    r=x.copy();d=angle.shape[-1];c=np.cos(angle);s=np.sin(angle)
    r[...,:2*d:2]=x[...,:2*d:2]*c+x[...,1:2*d:2]*s
    r[...,1:2*d:2]=-x[...,:2*d:2]*s+x[...,1:2*d:2]*c
    return r

def local_adjoint(inp,cot):
    assert inp.get('Input_States') is None
    z={k:np.asarray(v,dtype=np.float64) for k,v in inp.items() if v is not None}
    Q,K,V=(z[k] for k in ('Q','K','V'));B,T,Hq,N=Q.shape;H=V.shape[2];P=V.shape[-1]
    idx=np.arange(H)//(H//Hq);q=Q[:,:,idx]+z['Q_bias'][None,None];k=K[:,:,idx]+z['K_bias'][None,None]
    dt=z['DT'].transpose(0,2,1);a=np.exp(z['ADT'].transpose(0,2,1));lam=1/(1+np.exp(-z['Trap'].transpose(0,2,1)))
    beta=dt*(1-lam);gamma=dt*lam;tan=np.tanh(z['Angles']);delta=np.pi*dt[...,None]*tan;da=tan.shape[-1]
    state=np.zeros((B,H,P,N));zero=state.copy();states=[];hom=[];pred=[]
    for t in range(T):
        prev=V[:,t-1,:,:,None]*k[:,t-1,:,None,:] if t else zero
        g=a[:,t,:,None,None]*rowrot(state+beta[:,t,:,None,None]*prev,delta[:,t,:,None,:])
        state=g+gamma[:,t,:,None,None]*V[:,t,:,:,None]*k[:,t,:,None,:]
        states.append(state);hom.append(g);pred.append(np.sum(state*q[:,t,:,None,:],axis=-1))
    pre=np.stack(pred,axis=1)
    if 'D' in z:pre+=z['D'][None,None,:,None]*V
    sig=1/(1+np.exp(-z['Z'])) if 'Z' in z else np.ones_like(pre)
    gate=z['Z']*sig if 'Z' in z else np.ones_like(pre)
    dy=cot*gate;out=pre*gate;u=zero.copy();dq=np.zeros_like(q);dk=np.zeros_like(k)
    gr={k:np.zeros_like(v) for k,v in z.items()};ddt=np.zeros_like(dt);dadt=np.zeros_like(dt);dtrap=np.zeros_like(dt)
    if 'D' in z:gr['D']=np.sum(dy*V,axis=(0,1,3))
    if 'Z' in z:gr['Z']=cot*pre*sig*(1+z['Z']*(1-sig))
    for t in range(T-1,-1,-1):
        dq[:,t]=np.sum(states[t]*dy[:,t,:,:,None],axis=-2)
        p=u+dy[:,t,:,:,None]*q[:,t,:,None,:]
        future=beta[:,t+1,:,None,None]*u if t+1<T else zero
        c=gamma[:,t,:,None,None]*p+future
        dk[:,t]=np.sum(c*V[:,t,:,:,None],axis=-2)
        gr['V'][:,t]=np.sum(c*k[:,t,:,None,:],axis=-1)
        if 'D' in z:gr['V'][:,t]+=dy[:,t]*z['D'][None,:,None]
        gd=np.sum(p[...,:2*da:2]*hom[t][...,1:2*da:2]-p[...,1:2*da:2]*hom[t][...,:2*da:2],axis=-2)
        gr['Angles'][:,t]=np.pi*dt[:,t,:,None]*(1-tan[:,t]**2)*gd
        prev=V[:,t-1,:,:,None]*k[:,t-1,:,None,:] if t else zero
        prev=a[:,t,:,None,None]*rowrot(prev,delta[:,t,:,None,:]);cur=V[:,t,:,:,None]*k[:,t,:,None,:]
        ddt[:,t]=np.sum(p*((1-lam[:,t,:,None,None])*prev+lam[:,t,:,None,None]*cur),axis=(-2,-1))+np.pi*np.sum(tan[:,t]*gd,axis=-1)
        dadt[:,t]=np.sum(p*hom[t],axis=(-2,-1))
        dtrap[:,t]=dt[:,t]*lam[:,t]*(1-lam[:,t])*np.sum(p*(cur-prev),axis=(-2,-1))
        u=a[:,t,:,None,None]*rowrot(p,-delta[:,t,:,None,:])
    gr['DT']=ddt.transpose(0,2,1);gr['ADT']=dadt.transpose(0,2,1);gr['Trap']=dtrap.transpose(0,2,1)
    gr['Q_bias']=dq.sum(axis=(0,1));gr['K_bias']=dk.sum(axis=(0,1))
    gr['Q']=dq.reshape(B,T,Hq,H//Hq,N).sum(axis=3);gr['K']=dk.reshape(B,T,Hq,H//Hq,N).sum(axis=3)
    return out,gr

if __name__=='__main__':
    import torch,json,itertools
    from pathlib import Path
    from check_boundary_identity import inputs,sequential
    records=[]
    for T,(N,da),with_d,with_z in itertools.product((1,2,7,65),((4,2),(6,2)),(False,True),(False,True)):
        inp=inputs(T,N,da,False,91200+T)
        if not with_d:inp['D']=None
        if not with_z:inp['Z']=None
        out=sequential(**inp)[0];cot=torch.randn_like(out);(out*cot).sum().backward()
        ni={k:v.detach().numpy() if v is not None else None for k,v in inp.items()}
        y,g=local_adjoint(ni,cot.numpy())
        err=lambda x,r:float(np.max(np.abs(x-r))/max(1.,float(np.max(np.abs(r)))))
        errors={k:err(g[k],v.grad.numpy()) for k,v in inp.items() if v is not None}
        errors['forward']=err(y,out.detach().numpy())
        assert max(errors.values())<1e-10,errors
        records.append(dict(T=T,N=N,da=da,D=with_d,Z=with_z,errors=errors))
    summary=dict(status='PASS',cases=len(records),max_error=max(max(r['errors'].values()) for r in records),records=records)
    (VALIDATION/'local_rotation_validation.json').write_text(json.dumps(summary,indent=2))
    print('PASS',len(records),summary['max_error'])
