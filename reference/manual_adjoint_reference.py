"""Independent NumPy FP64 forward recurrence and hand-derived angle/DT adjoint.
No autograd, attention matrix, or boundary identity is used to compute gradients.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import *  # noqa: E402,F401,F403  (repository layout)
setup()
import numpy as np

def rotate(x,theta):
    result=x.copy();d=theta.shape[-1]
    c,s=np.cos(theta),np.sin(theta)
    result[...,:2*d:2]=x[...,:2*d:2]*c-x[...,1:2*d:2]*s
    result[...,1:2*d:2]=x[...,:2*d:2]*s+x[...,1:2*d:2]*c
    return result

def forward_adjoint(inp,cot,checkpoint_size=None):
    Q,K,V=[np.asarray(inp[k],dtype=np.float64) for k in ('Q','K','V')]
    B,T,Hq,N=Q.shape;H=V.shape[2];P=V.shape[-1];idx=np.arange(H)//(H//Hq)
    DT=np.asarray(inp['DT'],dtype=np.float64).transpose(0,2,1)
    a=np.exp(np.asarray(inp['ADT'],dtype=np.float64).transpose(0,2,1))
    lam=1/(1+np.exp(-np.asarray(inp['Trap'],dtype=np.float64).transpose(0,2,1)))
    beta=DT*(1-lam);gamma=DT*lam
    raw=np.asarray(inp['Angles'],dtype=np.float64);tan=np.tanh(raw);da=raw.shape[-1]
    theta=np.cumsum(np.pi*tan*DT[...,None],axis=1)
    st=inp.get('Input_States')
    if st is None:
        S=np.zeros((B,H,P,N));kp=np.zeros((B,H,N));vp=np.zeros((B,H,P))
    else:
        theta+=np.asarray(st[0],dtype=np.float64)[:,None]
        S,kp,vp=[np.asarray(x,dtype=np.float64) for x in st[1:]]
    initial=(S.copy(),kp.copy(),vp.copy())
    q=rotate(Q[:,:,idx]+np.asarray(inp['Q_bias'],dtype=np.float64)[None,None],theta)
    k=rotate(K[:,:,idx]+np.asarray(inp['K_bias'],dtype=np.float64)[None,None],theta)
    states=np.empty((T,B,H,P,N)) if checkpoint_size is None else None
    checkpoints={};out=np.empty((B,T,H,P))
    Z=inp.get('Z');gate=np.ones_like(out) if Z is None else np.asarray(Z,dtype=np.float64)/(1+np.exp(-np.asarray(Z,dtype=np.float64)))
    D=inp.get('D')
    for t in range(T):
        if checkpoint_size is not None and t%checkpoint_size==0:checkpoints[t]=S.copy()
        S=a[:,t,:,None,None]*(S+beta[:,t,:,None,None]*vp[..., :,None]*kp[...,None,:])
        S=S+gamma[:,t,:,None,None]*V[:,t,:,:,None]*k[:,t,:,None,:]
        if states is not None:states[t]=S
        y=np.einsum('bhpn,bhn->bhp',S,q[:,t])
        if D is not None:y=y+np.asarray(D)[None,:,None]*V[:,t]
        out[:,t]=y*gate[:,t]
        kp=k[:,t];vp=V[:,t]
    final=(out,np.remainder(theta[:,-1],2*np.pi),S,kp,vp)
    ds=np.asarray(cot[2],dtype=np.float64).copy()
    dk=np.zeros_like(k);dk[:,-1]=cot[3];dq=np.zeros_like(q);ddt=np.zeros_like(DT)
    loaded_start=-1;local_states=None
    for t in range(T-1,-1,-1):
        if checkpoint_size is not None:
            start=(t//checkpoint_size)*checkpoint_size
            if start!=loaded_start:
                stop=min(T,start+checkpoint_size);replay=checkpoints[start].copy()
                local_states=np.empty((stop-start,B,H,P,N))
                for j in range(start,stop):
                    prev_k=k[:,j-1] if j else initial[1];prev_v=V[:,j-1] if j else initial[2]
                    replay=a[:,j,:,None,None]*(replay+beta[:,j,:,None,None]*prev_v[..., :,None]*prev_k[...,None,:])
                    replay=replay+gamma[:,j,:,None,None]*V[:,j,:,:,None]*k[:,j,:,None,:]
                    local_states[j-start]=replay
                loaded_start=start
            state=local_states[t-start]
        else:state=states[t]
        dy=np.asarray(cot[0][:,t],dtype=np.float64)*gate[:,t]
        dq[:,t]=np.einsum('bhpn,bhp->bhn',state,dy)
        ds=ds+dy[..., :,None]*q[:,t,:,None,:]
        v=V[:,t];kv=k[:,t]
        kp=k[:,t-1] if t else initial[1];vp=V[:,t-1] if t else initial[2]
        dk[:,t]+=gamma[:,t,:,None]*np.einsum('bhpn,bhp->bhn',ds,v)
        if t:
            dk[:,t-1]+=beta[:,t,:,None]*a[:,t,:,None]*np.einsum('bhpn,bhp->bhn',ds,vp)
        ddt[:,t]=np.sum(ds*((1-lam[:,t,:,None,None])*a[:,t,:,None,None]*vp[..., :,None]*kp[...,None,:]
                            +lam[:,t,:,None,None]*v[..., :,None]*kv[...,None,:]),axis=(-2,-1))
        ds=a[:,t,:,None,None]*ds
    g=(-dq[...,:2*da:2]*q[...,1:2*da:2]+dq[...,1:2*da:2]*q[...,:2*da:2]
       -dk[...,:2*da:2]*k[...,1:2*da:2]+dk[...,1:2*da:2]*k[...,:2*da:2])
    g[:,-1]+=cot[1]
    suffix=np.flip(np.cumsum(np.flip(g,axis=1),axis=1),axis=1)
    dang=np.pi*DT[...,None]*(1-tan*tan)*suffix
    ddt+=np.pi*np.sum(tan*suffix,axis=-1)
    return final,dict(Angles=dang,DT=ddt.transpose(0,2,1),theta=g,suffix=suffix)

def self_test():
    import torch,json
    from pathlib import Path
    from check_boundary_identity import inputs,sequential,objective
    torch.set_num_threads(1);records=[]
    for T in (1,7,65):
        for initial in (False,True):
            for mask in (0,1,2,4,8,15):
                args=inputs(T,6,2,initial,8900+T+mask)
                out=sequential(**args)
                cot=[torch.randn_like(out[0])]+[torch.randn_like(out[i+1]) if mask&(1<<i) else torch.zeros_like(out[i+1]) for i in range(4)]
                objective(out,cot).backward()
                npin={k:(tuple(x.detach().numpy() for x in v) if isinstance(v,tuple) else v.detach().numpy() if v is not None else None) for k,v in args.items()}
                of,gr=forward_adjoint(npin,[x.numpy() for x in cot])
                er=lambda a,b:float(np.max(np.abs(a-b))/max(1.,float(np.max(np.abs(b)))))
                errors=[er(a,b.detach().numpy()) for a,b in zip(of,out)]+[er(gr[k],args[k].grad.numpy()) for k in ('Angles','DT')]
                assert max(errors)<1e-10,(T,initial,mask,errors)
                records.append(dict(T=T,initial=initial,mask=mask,max_error=max(errors)))
    result=dict(cases=len(records),max_error=max(r['max_error'] for r in records),status='PASS',records=records)
    (VALIDATION/'manual_adjoint_validation.json').write_text(json.dumps(result,indent=2))
    print(result['status'],result['cases'],result['max_error'])
if __name__=='__main__':self_test()
