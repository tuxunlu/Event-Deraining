import numpy as np
ROOT='/fs/nexus-scratch/tuxunlu/git/Event-Deraining/dataset/synthetic/synthetic_SPAC'

def auc(s,y):
    o=np.argsort(s,kind='mergesort'); r=np.empty(len(s)); r[o]=np.arange(1,len(s)+1)
    ss=s[o]; i=0
    while i<len(ss):
        j=i
        while j+1<len(ss) and ss[j+1]==ss[i]: j+=1
        if j>i: r[o[i:j+1]]=(i+j+2)/2.0
        i=j+1
    n1=y.sum(); n0=len(y)-n1
    if n1<5 or n0<5: return np.nan
    return (r[y==1].sum()-n1*(n1+1)/2.0)/(n1*n0)

def tube(x,y,t,v,dt=2.0):
    N=len(t); yw=np.floor(y-v*(t-t[0])).astype(np.int64)
    k=x*1000000+(yw+500000)
    o=np.lexsort((t,k)); ks=k[o]; ts=t[o]; c=np.zeros(N)
    st=np.flatnonzero(np.r_[True,ks[1:]!=ks[:-1]]); en=np.r_[st[1:],N]
    for s,e in zip(st,en):
        tt=ts[s:e]
        c[s:e]=np.searchsorted(tt,tt+dt,'right')-np.searchsorted(tt,tt-dt,'left')-1
    out=np.empty(N); out[o]=c; return out

def run(seq,idx,scale=1.0):
    r=np.load(f'{ROOT}/SPAC-dataset-merge/events/{seq}_Rain/{idx:010d}.npz')
    g=np.load(f'{ROOT}/SPAC-dataset-event/gt/{seq}_GT/{idx:010d}.npz')
    def key(d): return (((d['t'].astype(np.int64))*641+d['x'])*481+d['y'])*3+(d['p']>0)
    lab=np.isin(key(r),key(g)); israin=(~lab).astype(np.int64)
    x=(r['x']*scale).astype(np.int64); y=(r['y']*scale).astype(np.int64)
    t=r['t'].astype(np.float64)/1e6
    o=np.argsort(t,kind='mergesort'); x,y,t,israin=x[o],y[o],t[o],israin[o]
    print(f'--- {seq}[{idx}] N={len(t)} rainfrac={israin.mean():.3f} gtmatch={lab.sum()/len(g["t"]):.4f} scale={scale}')
    # per-pixel total count in window == what the collapsed frame knows
    pid=x*2000+y; cnt=np.bincount(pid); pixcnt=cnt[pid].astype(np.float64)
    print('  AUC[per-pixel count in window] (collapsed-frame proxy) =',round(auc(pixcnt,israin),4))
    vs=[-8,-4,-2,-1,-0.5,0,0.5,1,2,4,8]
    best=(0,0.5)
    for v in vs:
        c=tube(x,y,t,v)
        a=auc(c,israin);
        # conditional: residualise on per-pixel count via strata
        q=np.quantile(pixcnt,[.2,.4,.6,.8]); st=np.digitize(pixcnt,q)
        ac=[]; w=[]
        for s in range(5):
            m=st==s
            aa=auc(c[m],israin[m])
            if not np.isnan(aa): ac.append(aa); w.append(m.sum())
        cond=np.average(ac,weights=w) if ac else np.nan
        print(f'  v={v:>5}  AUC={a:.4f}  AUC|pixcount-strata={cond:.4f}')
        if abs(a-0.5)>abs(best[1]-0.5): best=(v,a)
    print('  best v',best)

for s,i in [('a1',5),('b2',5),('a3',5)]:
    try: run(s,i,1.0)
    except Exception as e: print('ERR',s,i,repr(e))
run('a1',5,256/640.)
