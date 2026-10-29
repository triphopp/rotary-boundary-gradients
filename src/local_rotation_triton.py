"""FP32 rotating-coordinate recurrence/direct adjoint: independent baseline.
Output-only, zero initial state, contiguous tensors, D and Z present.
Stores all FP32 states: explicitly trades memory for a simple accurate adjoint.
Computes every input gradient; no cumulative phase or reverse angular cumsum.
"""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _forward(Q,K,V,ADT,DT,Trap,QB,KB,Ang,D,Z,S,Y,
             T:tl.constexpr,H:tl.constexpr,HQ:tl.constexpr,N:tl.constexpr,P:tl.constexpr,DA:tl.constexpr,VP:tl.constexpr,NN:tl.constexpr):
    bh=tl.program_id(0);tile=tl.program_id(1);b=bh//H;h=bh%H;hq=h//(H//HQ)
    n=tl.arange(0,NN);p=tile*VP+tl.arange(0,VP)
    mask=(p[:,None]<P)&(n[None,:]<N//2)
    s0=tl.full((VP,NN),0,tl.float32);s1=s0
    kp0=tl.full((NN,),0,tl.float32);kp1=kp0;vp=tl.full((VP,),0,tl.float32)
    qb0=tl.load(QB+h*N+2*n,n<N//2,0).to(tl.float32);qb1=tl.load(QB+h*N+2*n+1,n<N//2,0).to(tl.float32)
    kb0=tl.load(KB+h*N+2*n,n<N//2,0).to(tl.float32);kb1=tl.load(KB+h*N+2*n+1,n<N//2,0).to(tl.float32)
    d=tl.load(D+h)
    for t in range(T):
        scalar=(b*H+h)*T+t;dt=tl.load(DT+scalar);a=tl.exp(tl.load(ADT+scalar));lam=tl.sigmoid(tl.load(Trap+scalar).to(tl.float32))
        raw=tl.load(Ang+((b*T+t)*H+h)*DA+n,n<DA,0)
        tan=2*tl.sigmoid(2*raw)-1;delta=3.141592653589793*dt*tan
        c=tl.cos(delta);sn=tl.sin(delta)
        off=((b*T+t)*HQ+hq)*N+2*n
        k0=tl.load(K+off,n<N//2,0).to(tl.float32)+kb0;k1=tl.load(K+off+1,n<N//2,0).to(tl.float32)+kb1
        q0=tl.load(Q+off,n<N//2,0).to(tl.float32)+qb0;q1=tl.load(Q+off+1,n<N//2,0).to(tl.float32)+qb1
        vo=((b*T+t)*H+h)*P+p;v=tl.load(V+vo,p<P,0).to(tl.float32)
        f0=s0+dt*(1-lam)*vp[:,None]*kp0[None,:];f1=s1+dt*(1-lam)*vp[:,None]*kp1[None,:]
        s0=a*(f0*c[None,:]+f1*sn[None,:])+dt*lam*v[:,None]*k0[None,:]
        s1=a*(-f0*sn[None,:]+f1*c[None,:])+dt*lam*v[:,None]*k1[None,:]
        so=(((b*T+t)*H+h)*P+p[:,None])*N+2*n[None,:]
        tl.store(S+so,s0,mask);tl.store(S+so+1,s1,mask)
        y=tl.sum(s0*q0[None,:]+s1*q1[None,:],axis=1)+d*v
        z=tl.load(Z+vo,p<P,0).to(tl.float32);y=y*z*tl.sigmoid(z)
        tl.store(Y+vo,y,p<P);kp0=k0;kp1=k1;vp=v

@tr.jit
def _backward(Q,K,V,ADT,DT,Trap,QB,KB,Ang,D,Z,S,Cot,
              PQ,PK,PA,PDT,PADT,PTR,PD,GV,GZ,
              T:tl.constexpr,H:tl.constexpr,HQ:tl.constexpr,N:tl.constexpr,P:tl.constexpr,DA:tl.constexpr,VP:tl.constexpr,NN:tl.constexpr,NT:tl.constexpr):
    bh=tl.program_id(0);tile=tl.program_id(1);b=bh//H;h=bh%H;hq=h//(H//HQ)
    n=tl.arange(0,NN);p=tile*VP+tl.arange(0,VP);mask=(p[:,None]<P)&(n[None,:]<N//2)
    u0=tl.full((VP,NN),0,tl.float32);u1=u0;bn=0.;dsum=0.
    qb0=tl.load(QB+h*N+2*n,n<N//2,0).to(tl.float32);qb1=tl.load(QB+h*N+2*n+1,n<N//2,0).to(tl.float32)
    kb0=tl.load(KB+h*N+2*n,n<N//2,0).to(tl.float32);kb1=tl.load(KB+h*N+2*n+1,n<N//2,0).to(tl.float32)
    d=tl.load(D+h)
    for step in range(T):
        t=T-1-step;scalar=(b*H+h)*T+t;dt=tl.load(DT+scalar);a=tl.exp(tl.load(ADT+scalar));lam=tl.sigmoid(tl.load(Trap+scalar).to(tl.float32))
        raw=tl.load(Ang+((b*T+t)*H+h)*DA+n,n<DA,0);tan=2*tl.sigmoid(2*raw)-1
        delta=3.141592653589793*dt*tan;c=tl.cos(delta);sn=tl.sin(delta)
        off=((b*T+t)*HQ+hq)*N+2*n
        k0=tl.load(K+off,n<N//2,0).to(tl.float32)+kb0;k1=tl.load(K+off+1,n<N//2,0).to(tl.float32)+kb1
        q0=tl.load(Q+off,n<N//2,0).to(tl.float32)+qb0;q1=tl.load(Q+off+1,n<N//2,0).to(tl.float32)+qb1
        kp0=tl.load(K+off-HQ*N,(n<N//2)&(t>0),0).to(tl.float32)+kb0
        kp1=tl.load(K+off-HQ*N+1,(n<N//2)&(t>0),0).to(tl.float32)+kb1
        vo=((b*T+t)*H+h)*P+p
        v=tl.load(V+vo,p<P,0).to(tl.float32);vp=tl.load(V+vo-H*P,(p<P)&(t>0),0).to(tl.float32)
        z=tl.load(Z+vo,p<P,0).to(tl.float32);sig=tl.sigmoid(z);cot=tl.load(Cot+vo,p<P,0).to(tl.float32);dy=cot*z*sig
        so=(((b*T+t)*H+h)*P+p[:,None])*N+2*n[None,:]
        s0=tl.load(S+so,mask,0);s1=tl.load(S+so+1,mask,0)
        sp0=tl.load(S+so-H*P*N,mask&(t>0),0);sp1=tl.load(S+so-H*P*N+1,mask&(t>0),0)
        prev0=vp[:,None]*kp0[None,:];prev1=vp[:,None]*kp1[None,:]
        f0=sp0+dt*(1-lam)*prev0;f1=sp1+dt*(1-lam)*prev1
        g0=a*(f0*c[None,:]+f1*sn[None,:]);g1=a*(-f0*sn[None,:]+f1*c[None,:])
        p0=u0+dy[:,None]*q0[None,:];p1=u1+dy[:,None]*q1[None,:]
        cc0=dt*lam*p0+bn*u0;cc1=dt*lam*p1+bn*u1
        po=(((b*T+t)*H+h)*NT+tile)*N+2*n
        tl.store(PQ+po,tl.sum(s0*dy[:,None],axis=0),n<N//2);tl.store(PQ+po+1,tl.sum(s1*dy[:,None],axis=0),n<N//2)
        tl.store(PK+po,tl.sum(cc0*v[:,None],axis=0),n<N//2);tl.store(PK+po+1,tl.sum(cc1*v[:,None],axis=0),n<N//2)
        tl.store(GV+vo,tl.sum(cc0*k0[None,:]+cc1*k1[None,:],axis=1)+dy*d,p<P)
        pre=tl.sum(s0*q0[None,:]+s1*q1[None,:],axis=1)+d*v
        tl.store(GZ+vo,cot*pre*sig*(1+z*(1-sig)),p<P)
        gd=tl.sum(p0*g1-p1*g0,axis=0)
        ao=(((b*T+t)*H+h)*NT+tile)*DA+n
        tl.store(PA+ao,3.141592653589793*dt*(1-tan*tan)*gd,n<DA)
        pr0=a*(prev0*c[None,:]+prev1*sn[None,:]);pr1=a*(-prev0*sn[None,:]+prev1*c[None,:])
        cur0=v[:,None]*k0[None,:];cur1=v[:,None]*k1[None,:]
        ddt=tl.sum(tl.sum(p0*((1-lam)*pr0+lam*cur0)+p1*((1-lam)*pr1+lam*cur1),axis=1),axis=0)+3.141592653589793*tl.sum(tl.where(n<DA,tan*gd,0),axis=0)
        dadt=tl.sum(tl.sum(p0*g0+p1*g1,axis=1),axis=0)
        dtrap=dt*lam*(1-lam)*tl.sum(tl.sum(p0*(cur0-pr0)+p1*(cur1-pr1),axis=1),axis=0)
        scalarout=((b*T+t)*H+h)*NT+tile
        tl.store(PDT+scalarout,ddt);tl.store(PADT+scalarout,dadt);tl.store(PTR+scalarout,dtrap)
        dsum+=tl.sum(dy*v,axis=0)
        u0=a*(p0*c[None,:]-p1*sn[None,:]);u1=a*(p0*sn[None,:]+p1*c[None,:]);bn=dt*(1-lam)
    tl.store(PD+bh*NT+tile,dsum)

def forward(inp,vp=8):
    assert inp.get('Input_States') is None and inp['D'] is not None and inp['Z'] is not None
    assert all(x.is_contiguous() for x in inp.values() if x is not None)
    B,T,HQ,N=inp['Q'].shape;_,_,H,P=inp['V'].shape;DA=inp['Angles'].shape[-1]
    assert N%2==0 and DA<=N//2
    states=torch.empty((B,T,H,P,N),device='cuda',dtype=torch.float32);out=torch.empty_like(inp['V'])
    _forward[(B*H,tr.cdiv(P,vp))](*[inp[k] for k in ('Q','K','V','ADT','DT','Trap','Q_bias','K_bias','Angles','D','Z')],states,out,T,H,HQ,N,P,DA,vp,tr.next_power_of_2(N//2),num_warps=4,enable_fp_fusion=False)
    return out,states

def backward(inp,states,cot,vp=8):
    B,T,HQ,N=inp['Q'].shape;_,_,H,P=inp['V'].shape;DA=inp['Angles'].shape[-1];nt=tr.cdiv(P,vp)
    new=lambda shape:torch.empty(shape,device='cuda',dtype=torch.float32)
    pq=new((B,T,H,nt,N));pk=torch.empty_like(pq);pa=new((B,T,H,nt,DA));pdt=new((B,T,H,nt));padt=torch.empty_like(pdt);ptr=torch.empty_like(pdt);pd=new((B,H,nt))
    gv=torch.empty_like(inp['V']);gz=torch.empty_like(inp['Z'])
    _backward[(B*H,nt)](*[inp[k] for k in ('Q','K','V','ADT','DT','Trap','Q_bias','K_bias','Angles','D','Z')],states,cot,pq,pk,pa,pdt,padt,ptr,pd,gv,gz,T,H,HQ,N,P,DA,vp,tr.next_power_of_2(N//2),nt,num_warps=4,enable_fp_fusion=False)
    dq=pq.sum(3);dk=pk.sum(3)
    gr=dict(Q=dq.reshape(B,T,HQ,H//HQ,N).sum(3),K=dk.reshape(B,T,HQ,H//HQ,N).sum(3),
            Q_bias=dq.sum((0,1)),K_bias=dk.sum((0,1)),V=gv,Z=gz,D=pd.sum((0,2)),
            Angles=pa.sum(3),DT=pdt.sum(3).transpose(1,2),ADT=padt.sum(3).transpose(1,2),Trap=ptr.sum(3).transpose(1,2))
    return {k:g.to(inp[k].dtype) for k,g in gr.items()}
