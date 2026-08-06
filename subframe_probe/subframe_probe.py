import numpy as np, glob, os
ROOT='/fs/nexus-scratch/tuxunlu/git/Event-Deraining/dataset/synthetic/synthetic_SPAC'
def load(p):
    d=np.load(p,allow_pickle=True); return d['t'],d['x'],d['y'],d['p']
def key(t,x,y,p):
    return (t.astype(np.int64)<<24) ^ (x.astype(np.int64)<<12) ^ (y.astype(np.int64)<<2) ^ ((p>0).astype(np.int64))
seqs=sorted(os.listdir(f'{ROOT}/SPAC-dataset-merge/events'))
print('seqs',seqs)
tot=0; totr=0
for s in seqs[:3]:
    g=s.replace('_Rain','_GT')
    fs=sorted(glob.glob(f'{ROOT}/SPAC-dataset-merge/events/{s}/*.npz'))[:3]
    for f in fs:
        gf=f.replace('SPAC-dataset-merge/events','SPAC-dataset-event/gt').replace(s,g)
        if not os.path.exists(gf): print('missing',gf); continue
        t,x,y,p=load(f); tg,xg,yg,pg=load(gf)
        km=key(t,x,y,p); kg=set(key(tg,xg,yg,pg).tolist())
        isbg=np.array([k in kg for k in km.tolist()])
        tot+=len(t); totr+=int((~isbg).sum())
        print(f'{s} {os.path.basename(f)} N={len(t)} gt={len(tg)} matched_bg={isbg.sum()} rain={(~isbg).sum()} gt_recovered={isbg.sum()/max(len(tg),1):.4f}')
print('TOTAL',tot,'rain frac',totr/tot)
