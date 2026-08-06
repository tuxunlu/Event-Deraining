import numpy as np, glob, os
ROOT='/fs/nexus-scratch/tuxunlu/git/Event-Deraining/dataset/synthetic/synthetic_SPAC'
def load(p):
    d=np.load(p,allow_pickle=True); return d['t'],d['x'],d['y'],d['p']
def key(t,x,y,p):
    return (t.astype(np.int64)<<24)^(x.astype(np.int64)<<12)^(y.astype(np.int64)<<2)^((p>0).astype(np.int64))
print("=== CONFOUND CHECK: video-frame quantization of timestamps ===")
for s in ['a1_Rain','a3_Rain','b1_Rain']:
    g=s.replace('_Rain','_GT')
    f=sorted(glob.glob(f'{ROOT}/SPAC-dataset-merge/events/{s}/*.npz'))[1]
    gf=f.replace('SPAC-dataset-merge/events','SPAC-dataset-event/gt').replace(s,g)
    t,x,y,p=load(f); tg,xg,yg,pg=load(gf)
    kg=set(key(tg,xg,yg,pg).tolist()); km=key(t,x,y,p)
    rain=np.array([0 if k in kg else 1 for k in km.tolist()],dtype=bool)
    t0=t-t.min()
    for nm,per in [('33.3ms(30fps)',33333333),('41.7ms(24fps)',41666667)]:
        ph_r=(t0[rain]%per)/per; ph_b=(t0[~rain]%per)/per
        # uniformity: max deviation of 20-bin histogram from uniform
        hr,_=np.histogram(ph_r,20,(0,1)); hb,_=np.histogram(ph_b,20,(0,1))
        print(f'{s} {nm}: rain phase nonunif={np.abs(hr/hr.sum()-0.05).max():.4f} bg={np.abs(hb/hb.sum()-0.05).max():.4f}')
    # distinct-timestamp collision: do rain events share timestamps (batch-rendered)?
    ur=len(np.unique(t[rain]))/rain.sum(); ub=len(np.unique(t[~rain]))/(~rain).sum()
    print(f'   unique-ts ratio  rain={ur:.4f}  bg={ub:.4f}   (1.0=all distinct)')
    d=np.diff(np.sort(t[rain])); print(f'   rain inter-event dt ns: p50={np.median(d):.0f} p10={np.percentile(d,10):.0f}')
    d2=np.diff(np.sort(t[~rain])); print(f'   bg   inter-event dt ns: p50={np.median(d2):.0f} p10={np.percentile(d2,10):.0f}')
