"""Fit an offline short-horizon return-advantage predictor."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
import torch.nn.functional as F
from supermimic.strategic_critic import StrategicErrorCritic

ap=argparse.ArgumentParser(); ap.add_argument('dataset',type=Path); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--epochs',type=int,default=100); ap.add_argument('--batch',type=int,default=256); ap.add_argument('--hidden',type=int,default=256); ap.add_argument('--lr',type=float,default=3e-4); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--seed',type=int,default=0)
a=ap.parse_args(); torch.manual_seed(a.seed); dev=torch.device(a.device); d=torch.load(a.dataset,map_location=dev,weights_only=False)
s,act,xi,y=(d[k].float().to(dev) for k in ('s','a','xi','advantage')); model=StrategicErrorCritic(hidden=a.hidden).to(dev); x0=model.features(s,act,xi)
perm=torch.randperm(len(y),device=dev); nv=max(1,len(y)//5); vi,ti=perm[:nv],perm[nv:]; mean=x0[ti].mean(0); std=x0[ti].std().clamp_min(1e-3); x=(x0-mean)/std
ym,ys=y[ti].mean(),y[ti].std().clamp_min(1e-3); yn=(y-ym)/ys; opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
for _ in range(a.epochs):
 order=ti[torch.randperm(len(ti),device=dev)]
 for j in range(0,len(order),a.batch):
  idx=order[j:j+a.batch]; loss=F.smooth_l1_loss(model(x[idx]),yn[idx]); opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
 pred=model(x)*ys+ym; rmse=lambda z: float(((pred[z]-y[z])**2).mean().sqrt()); corr=float(torch.corrcoef(torch.stack((pred[vi],y[vi])))[0,1])
out={'samples':len(y),'train_rmse':rmse(ti),'val_rmse':rmse(vi),'val_corr':corr,'horizon':d.get('metadata',{}).get('horizon')}; print(json.dumps(out,indent=2)); a.output.parent.mkdir(parents=True,exist_ok=True); torch.save({'model':model.state_dict(),'mean':mean.cpu(),'std':std.cpu(),'target_mean':ym.cpu(),'target_std':ys.cpu(),'hidden':a.hidden,'metadata':out},a.output)
