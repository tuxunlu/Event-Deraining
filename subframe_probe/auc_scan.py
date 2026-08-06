import numpy as np, glob, os
ROOT='/fs/nexus-scratch/tuxunlu/git/Event-Deraining/dataset/synthetic/synthetic_SPAC'
H,W=480,640; M=4; R=2  # 5x5 nbhd
NEG=np.int64(-1<<62)
TAUS=[('31us',31250),('100us',100000),('300us',300000),('1ms',1000000),('3ms',3000000),
      ('10ms',10000000),('30ms',30000000),('104ms',104000000)]
def load(p):
    d=np.load(p,allow_pickle=True); return d['t'],d['x'],d['y'],d['p']
def key(t,x,y,p):
    return (t.astype(np.int64)<<24)^(x.astype(np.int64)<<12)^(y.astype(np.int64)<<2)^((p>0).astype(np.int64))
def auc(sc,lab):
    # lab: 1 = rain (positive)
    ok=np.isfinite(sc); sc=sc[ok]; lab=lab[ok]
    if lab.sum()==0 or (1-lab).sum()==0: return np.nan
    r=np.argsort(np.argsort(sc))+1.0
    n1=lab.sum(); n0=len(lab)-n1
    return (r[lab==1].sum()-n1*(n1+1)/2)/(n1*n0)

feats={}; labs=[]
seqs=['a1_Rain','a2_Rain','a3_Rain','b1_Rain','b2_Rain']
for s in seqs:
    g=s.replace('_Rain','_GT')
    for f in sorted(glob.glob(f'{ROOT}/SPAC-dataset-merge/events/{s}/*.npz'))[:2]:
        gf=f.replace('SPAC-dataset-merge/events','SPAC-dataset-event/gt').replace(s,g)
        if not os.path.exists(gf): continue
        t,x,y,p=load(f); tg,xg,yg,pg=load(gf)
        o=np.argsort(t,kind='stable'); t,x,y,p=t[o],x[o],y[o],p[o]
        kg=set(key(tg,xg,yg,pg).tolist()); km=key(t,x,y,p)
        rain=np.array([0 if k in kg else 1 for k in km.tolist()],dtype=np.int8)
        last=np.full((M,H+2*R,W+2*R),NEG,dtype=np.int64)   # recent ts per pixel
        N=len(t)
        f_self=np.empty(N); f_nbr=np.empty(N); f_cnt={n:np.empty(N) for n,_ in TAUS}
        for i in range(N):
            ti=t[i]; xi=x[i]+R; yi=y[i]+R
            win=last[:,yi-R:yi+R+1,xi-R:xi+R+1]
            f_self[i]=ti-last[0,yi,xi]
            mx=win.max()
            f_nbr[i]=ti-mx
            wf=win.ravel()
            for n,tau in TAUS:
                f_cnt[n][i]=np.count_nonzero(wf>=ti-tau)
            last[1:,yi,xi]=last[:-1,yi,xi]; last[0,yi,xi]=ti
        labs.append(rain)
        feats.setdefault('dt_self',[]).append(f_self)
        feats.setdefault('dt_nbr5',[]).append(f_nbr)
        for n,_ in TAUS: feats.setdefault('cnt5_'+n,[]).append(f_cnt[n])
        print('done',s,os.path.basename(f),N,flush=True)
lab=np.concatenate(labs)
print('\nN events',len(lab),'rain frac',lab.mean())
print('\n=== per-event AUC (rain=positive), 0.5=chance ===')
for k in ['dt_self','dt_nbr5']+['cnt5_'+n for n,_ in TAUS]:
    v=np.concatenate(feats[k]).astype(np.float64)
    v=np.where(v>1e17,np.nan,v)
    a=auc(v,lab.astype(int))
    print(f'{k:14s} AUC={a:.4f}  (|AUC-0.5|={abs(a-0.5):.4f})')
np.savez('/tmp/auc_feats.npz',lab=lab,**{k:np.concatenate(v) for k,v in feats.items()})
