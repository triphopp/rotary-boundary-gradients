"""Independent O(T) FP64 recurrence vs the project's O(T^2) reference.
No Triton code or existing boundary implementation is reused.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import *  # noqa: E402,F401,F403  (repository layout)
setup()
import json, math, sys, itertools
from pathlib import Path
import torch
from ref_torch64 import mamba3_siso_ref64
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

def rotate(x,theta):
    pairs=x.reshape(*x.shape[:-1],-1,2)
    a,b=pairs.unbind(-1)
    if theta.shape[-1]<a.shape[-1]:
        theta=torch.cat((theta,theta.new_zeros(*theta.shape[:-1],a.shape[-1]-theta.shape[-1])),-1)
    c,s=theta.cos(),theta.sin()
    return torch.stack((c*a-s*b,s*a+c*b),-1).flatten(-2)

def sequential(Q,K,V,ADT,DT,Trap,Q_bias,K_bias,Angles,D=None,Z=None,Input_States=None,keep=None,theta_override=None):
    B,T,Hq,N=Q.shape;H=V.shape[2];P=V.shape[-1];da=Angles.shape[-1]
    idx=torch.arange(H)//(H//Hq)
    qraw=Q[:,:,idx]+Q_bias[None,None]
    kraw=K[:,:,idx]+K_bias[None,None]
    theta=torch.cumsum(torch.tanh(Angles)*math.pi*DT.transpose(1,2)[...,None],1)
    if Input_States is not None:theta=theta+Input_States[0][:,None]
    if theta_override is not None:theta=theta_override
    if keep is not None:
        theta.retain_grad();keep['theta']=theta;keep['folded']=[]
    q,k=rotate(qraw,theta),rotate(kraw,theta)
    if Input_States is None:
        S=Q.new_zeros(B,H,P,N);kp=Q.new_zeros(B,H,N);vp=Q.new_zeros(B,H,P)
    else:_,S,kp,vp=Input_States
    ys=[]
    for t in range(T):
        lam=torch.sigmoid(Trap[:,:,t]);beta=DT[:,:,t]*(1-lam);gamma=DT[:,:,t]*lam
        folded=S+beta[...,None,None]*vp[..., :,None]*kp[...,None,:]
        if keep is not None:folded.retain_grad();keep['folded'].append(folded)
        S=ADT[:,:,t].exp()[...,None,None]*folded+gamma[...,None,None]*V[:,t,:,:,None]*k[:,t,:,None,:]
        y=(S*q[:,t,:,None,:]).sum(-1)
        if D is not None:y=y+D[None,:,None]*V[:,t]
        if Z is not None:y=y*Z[:,t]*torch.sigmoid(Z[:,t])
        ys.append(y);kp=k[:,t];vp=V[:,t]
    return torch.stack(ys,1),theta[:,-1].remainder(2*math.pi),S,kp,vp

def inputs(T,N,da,initial,seed):
    torch.manual_seed(seed);B,H,Hq,P=1,2,1,3
    r=lambda *s: (.2*torch.randn(*s)).requires_grad_()
    args=dict(Q=r(B,T,Hq,N),K=r(B,T,Hq,N),V=r(B,T,H,P),
              ADT=(-.03-.07*torch.rand(B,H,T)).requires_grad_(),DT=(.1+.2*torch.rand(B,H,T)).requires_grad_(),
              Trap=r(B,H,T),Q_bias=r(H,N),K_bias=r(H,N),Angles=r(B,T,H,da),D=r(H),Z=r(B,T,H,P))
    args['Input_States']=(r(B,H,da),r(B,H,P,N),r(B,H,N),r(B,H,P)) if initial else None
    return args

def flatten(args):
    return [x for v in args.values() if v is not None for x in (v if isinstance(v,tuple) else (v,))]
def clone(args):
    return {k:(tuple(x.detach().clone().requires_grad_() for x in v) if isinstance(v,tuple) else
               v.detach().clone().requires_grad_() if v is not None else None) for k,v in args.items()}
def objective(out,cot):return sum((x*c).sum() for x,c in zip(out,cot))
def relative(a,b):return float((a-b).abs().max()/max(1.,float(b.abs().max())))
def sj(x,da):return torch.stack((x[...,1:2*da:2],-x[...,0:2*da:2]),-1).flatten(-2)

def main():
    records=[];fd=[];worst_wrong=0.
    for T,initial,(N,da),mask in itertools.product((1,2,7,65),(False,True),((4,2),(6,2)),range(16)):
        args=inputs(T,N,da,initial,7100+T+mask);keep={}
        out=sequential(**args,keep=keep)
        cot=[torch.randn_like(out[0])]+[torch.randn_like(out[i+1]) if mask&(1<<i) else torch.zeros_like(out[i+1]) for i in range(4)]
        loss=objective(out,cot);loss.backward()
        grad_seq=[x.grad.clone() if x.grad is not None else torch.zeros_like(x) for x in flatten(args)]
        refargs=clone(args);refout=mamba3_siso_ref64(**refargs)
        objective(refout,cot).backward()
        grad_ref=[x.grad if x.grad is not None else torch.zeros_like(x) for x in flatten(refargs)]
        forward_error=max(relative(a,b) for a,b in zip(out,refout))
        gradient_error=max(relative(a,b) for a,b in zip(grad_seq,grad_ref))
        term=-((cot[2][...,:2*da]*sj(out[2],da)).reshape(1,2,3,da,2)).sum((-3,-1))
        term=term-(cot[3][...,:2*da]*sj(out[3],da)).reshape(1,2,da,2).sum(-1)+cot[1]
        boundary=[]
        for state in keep['folded']:
            boundary.append((state.grad[...,:2*da]*sj(state,da)).reshape(1,2,3,da,2).sum((-3,-1)))
        direct=torch.stack(boundary,1)+term[:,None]
        suffix=keep['theta'].grad.flip(1).cumsum(1).flip(1)
        error=relative(direct,suffix)
        wrong=relative(torch.stack(boundary,1),suffix);worst_wrong=max(worst_wrong,wrong)
        records.append(dict(T=T,initial=initial,N=N,da=da,terminal_mask=mask,
                            forward_error=forward_error,gradient_error=gradient_error,identity_error=error,
                            missing_terminal_error=wrong))
        if T==7 and mask in (0,15):
            u=3;theta=keep['theta'].detach();direction=torch.zeros_like(theta);direction[:,u:]=1
            expected=float(suffix[:,u].sum())
            for eps in (1e-4,1e-5,1e-6):
                with torch.no_grad():
                    plus=objective(sequential(**args,theta_override=theta+eps*direction),cot)
                    minus=objective(sequential(**args,theta_override=theta-eps*direction),cot)
                    value=float((plus-minus)/(2*eps))
                fd.append(dict(initial=initial,N=N,mask=mask,eps=eps,error=abs(value-expected)/max(1,abs(expected))))
        assert max(error,forward_error,gradient_error)<1e-10,records[-1]
    assert max(r['error'] for r in fd)<1e-7
    summary=dict(cases=len(records),boundary_checks=sum(r['T'] for r in records),
                 max_identity_error=max(r['identity_error'] for r in records),
                 max_forward_error=max(r['forward_error'] for r in records),
                 max_gradient_error=max(r['gradient_error'] for r in records),
                 finite_difference_checks=len(fd),max_fd_error=max(r['error'] for r in fd),
                 max_error_if_terminal_omitted=worst_wrong,status='PASS')
    (VALIDATION/'boundary_identity_results.json').write_text(json.dumps(dict(summary=summary,cases=records,finite_differences=fd),indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
