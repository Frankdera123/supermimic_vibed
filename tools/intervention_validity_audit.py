"""Held-out validity audit for a simulator-trained intervention adapter."""
from __future__ import annotations
import argparse,json,sys
import torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from supermimic.networks import WorldModel
from supermimic.mdp import unflatten_state,normalize_xi
from supermimic.rotations import so3_exp,yaw_mat

def apply_delta(S,z):
 p,R,v,w,q,qd=S; H=yaw_mat(R); dp,dth,dv,dw,dq,dqd=z.split((3,3,3,3,29,29),-1)
 return (p+(H@dp.unsqueeze(-1)).squeeze(-1),R@so3_exp(dth),v+(R@dv.unsqueeze(-1)).squeeze(-1),w+(R@dw.unsqueeze(-1)).squeeze(-1),q+dq,qd+dqd)
def rows(path,wm,adapter,mean,std,blend,dev):
 d=torch.load(path,map_location=dev,weights_only=False); s0f=d['pairs'][0]['s0'].to(dev); S=unflatten_state(s0f); xi=normalize_xi(d['xi'].to(dev)); act=d['action'].to(dev); dirs=d['directions'].to(dev); X=[];Y=[]
 for pair in d['pairs']:
  delta=float(pair['delta'])
  for k in range(dirs.shape[1]):
   for sign,key in ((1,'plus'),(-1,'minus')):
    a=act+sign*delta*dirs[:,k]; pred=wm.step(S,a,xi); pd=wm.state_error(pred,S); feat=torch.cat((s0f,a,pd),-1); corr=pd+blend*(adapter((feat-mean)/std)-pd); cp=apply_delta(S,corr); true=unflatten_state(pair[key].to(dev)[:,k]); sc=torch.maximum(wm.out_norm.std(),torch.full_like(wm.out_norm.std(),1e-4)); e0=(wm.state_error(pred,true)/sc).square().mean(-1); e1=(wm.state_error(cp,true)/sc).square().mean(-1); X.append(feat.detach()); Y.append((e1<e0).float())
 return torch.cat(X),torch.cat(Y)
def auc(score,y):
 order=torch.argsort(score); yy=y[order]; n1=yy.sum(); n0=len(y)-n1
 if n1==0 or n0==0:return .5
 ranks=torch.arange(1,len(y)+1,device=y.device); return float(((ranks*yy).sum()-n1*(n1+1)/2)/(n1*n0))
ap=argparse.ArgumentParser(); ap.add_argument('adapter'); ap.add_argument('train_dataset'); ap.add_argument('eval_datasets',nargs='+'); ap.add_argument('--wm_checkpoint',required=True); ap.add_argument('--blend',type=float,default=.1); ap.add_argument('--epochs',type=int,default=200); ap.add_argument('--quantile',type=float,default=.75); ap.add_argument('--device',default='cuda:0'); a=ap.parse_args(); dev=a.device
ack=torch.load(a.adapter,map_location=dev,weights_only=False); adapter=nn.Sequential(nn.Linear(170,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,70)).to(dev); adapter.load_state_dict(ack['model']); adapter.eval(); mean=ack['input_mean'].to(dev); std=ack['input_std'].to(dev).clamp_min(1e-4)
wck=torch.load(a.wm_checkpoint,map_location=dev,weights_only=False); h=int(wck['wm']['residual_net.0.weight'].shape[0]) if 'residual_net.0.weight' in wck['wm'] else 0; wm=WorldModel(residual_hidden=h).to(dev); wm.load_state_dict(wck['wm']); wm.eval()
X,y=rows(a.train_dataset,wm,adapter,mean,std,a.blend,dev); clf=nn.Sequential(nn.Linear(170,64),nn.SiLU(),nn.Linear(64,1)).to(dev); opt=torch.optim.AdamW(clf.parameters(),lr=3e-3,weight_decay=1e-4); pos=y.sum().clamp_min(1); neg=len(y)-pos; wt=torch.where(y>0.5,neg/pos,torch.ones_like(y))
for _ in range(a.epochs):
 loss=nn.functional.binary_cross_entropy_with_logits(clf((X-mean)/std).squeeze(-1),y,weight=wt); opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad(): score=clf((X-mean)/std).squeeze(-1).sigmoid(); th=torch.quantile(score,a.quantile)
out={'train':a.train_dataset,'train_positive':float(y.mean()),'results':[]}
for p in a.eval_datasets:
 Xe,ye=rows(p,wm,adapter,mean,std,a.blend,dev)
 with torch.no_grad(): se=clf((Xe-mean)/std).squeeze(-1).sigmoid()
 sel=se>=th; out['results'].append({'dataset':p,'positive_rate':float(ye.mean()),'auc':auc(se,ye),'selected_rate':float(sel.float().mean()),'selected_precision':float(ye[sel].mean()) if sel.any() else 0.0})
print(json.dumps(out,indent=2))
