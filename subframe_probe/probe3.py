import numpy as np, glob, os
ROOT='/fs/nexus-scratch/tuxunlu/git/Event-Deraining/dataset/synthetic/synthetic_SPAC'
H,W=480,640; R=2; NEG=np.int64(-1<<62)
def load(p):
    d=np.load(p,allow_pickle=True); return d['t'],d['x'],d['y'],d['p']
def key(t,x,y,p):
    return (t.astype(np.int64)<<24)^(x.astype(np.int64)<<12)^(y.astype(np.int64)<<2)^((p>0).astype(np.int64))
def auc(sc,lab):
    ok=np.isfinite(sc); sc,lab=sc[ok],lab[ok]
    r=np.argsort(np.argsort(sc))+1.0; n1=lab.sum(); n0=len(lab)-n1
    return (r[lab==1].sum()-n1*(n1+1)/2)/(n1*n0)
F={}; L=[]
for s in ['a1_Rain','a2_Rain','a3_Rain','b1_Rain','b2_Rain','b3_Rain']:
    g=s.replace('_Rain','_GT')
    for f in sorted(glob.glob(f'{ROOT}/SPAC-dataset-merge/events/{s}/*.npz'))[:2]:
        gf=f.replace('SPAC-dataset-merge/events','SPAC-dataset-event/gt').replace(s,g)
        if not os.path.exists(gf): continue
        t,x,y,p=load(f); tg,xg,yg,pg=load(gf)
        o=np.argsort(t,kind='stable'); t,x,y,p=t[o],x[o],y[o],p[o]
        kg=set(key(tg,xg,yg,pg).tolist())
        rain=np.array([0 if k in kg else 1 for k in key(t,x,y,p).tolist()],dtype=np.int8)
        # LEAKAGE feature: how many events share this exact timestamp
        uq,inv,cnt=np.unique(t,return_inverse=True,return_counts=True)
        share=cnt[inv].astype(float)
        N=len(t); last=np.full((H+2*R,W+2*R),NEG,dtype=np.int64)
        f_up=np.empty(N); f_dn=np.empty(N); f_lr=np.empty(N); f_self=np.empty(N); f_c3=np.empty(N)
        for i in range(N):
            ti=t[i]; xi=x[i]+R; yi=y[i]+R
            up=last[yi-2:yi,xi-1:xi+2].max(); dn=last[yi+1:yi+3,xi-1:xi+2].max()
            lr=max(last[yi,xi-2:xi].max(),last[yi,xi+1:xi+3].max())
            f_up[i]=ti-up; f_dn[i]=ti-dn; f_lr[i]=ti-lr; f_self[i]=ti-last[yi,xi]
            f_c3[i]=np.count_nonzero(last[yi-R:yi+R+1,xi-R:xi+R+1]>=ti-3000000)
            last[yi,xi]=ti
        cl=lambda a:np.where(a>1e17,np.nan,a)
        F.setdefault('share_ts',[]).append(share)
        F.setdefault('dt_self',[]).append(cl(f_self)); F.setdefault('dt_up',[]).append(cl(f_up))
        F.setdefault('dt_dn',[]).append(cl(f_dn)); F.setdefault('dt_lr',[]).append(cl(f_lr))
        F.setdefault('cnt5_3ms',[]).append(f_c3)
        F.setdefault('vert_asym',[]).append(cl(np.log1p(np.abs(f_dn))-np.log1p(np.abs(f_up))))
        L.append(rain)
lab=np.concatenate(L).astype(int)
print('N',len(lab),'rain frac',round(lab.mean(),4))
print('\n=== single-feature AUC (rain=positive) ===')
for k in F: print(f'  {k:12s} AUC={auc(np.concatenate(F[k]).astype(float),lab):.4f}')
# combined classifier, grouped train/test split by file
X=np.column_stack([np.nan_to_num(np.concatenate(F[k]).astype(float),nan=-1,posinf=1e12) for k in F])
X=np.sign(X)*np.log1p(np.abs(X))
n=len(lab); idx=np.arange(n); tr=idx[:int(.6*n)]; te=idx[int(.6*n):]
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
m=HistGradientBoostingClassifier(max_iter=120,max_depth=6).fit(X[tr],lab[tr])
pr=m.predict_proba(X[te])[:,1]
print(f'\n=== COMBINED sub-frame features ===\n  test AUC={roc_auc_score(lab[te],pr):.4f}')
best=max(((0.5*((pr>=th)[lab[te]==1].mean()+(pr<th)[lab[te]==0].mean()),th) for th in np.quantile(pr,np.linspace(.01,.99,99))))
print(f'  best balanced acc (DA-equivalent) = {best[0]:.4f}')
# WITHOUT the leakage feature
ks=[k for k in F if k!='share_ts']
X2=np.column_stack([np.nan_to_num(np.concatenate(F[k]).astype(float),nan=-1,posinf=1e12) for k in ks])
X2=np.sign(X2)*np.log1p(np.abs(X2))
m2=HistGradientBoostingClassifier(max_iter=120,max_depth=6).fit(X2[tr],lab[tr])
pr2=m2.predict_proba(X2[te])[:,1]
b2=max(((0.5*((pr2>=th)[lab[te]==1].mean()+(pr2<th)[lab[te]==0].mean()),th) for th in np.quantile(pr2,np.linspace(.01,.99,99))))
print(f'\n=== COMBINED, LEAKAGE FEATURE REMOVED ===\n  test AUC={roc_auc_score(lab[te],pr2):.4f}\n  best balanced acc = {b2[0]:.4f}')
