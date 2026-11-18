"""Generate +/- local action-gradient candidates from a WM checkpoint."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from supermimic.mdp import normalize_xi, unflatten_state
from supermimic.networks import WorldModel

ap=argparse.ArgumentParser(); ap.add_argument('checkpoint',type=Path); ap.add_argument('dataset',type=Path); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--epsilon',type=float,default=.005); ap.add_argument('--limit',type=int,default=256); ap.add_argument('--motion_dir',default=None); ap.add_argument('--device',default='cuda:0')
a=ap.parse_args(); dev=a.device; ck=torch.load(a.checkpoint,map_location=dev,weights_only=False); d=torch.load(a.dataset,map_location=dev,weights_only=False)
state=d['s'].to(dev); act=d['a'].to(dev); xi=d['xi'].to(dev); f=d['f'].to(dev); a_prev=d.get('a_prev')
if a_prev is not None: a_prev=a_prev.to(dev)
groups=int(d.get('metadata',{}).get('action_candidates',4)); state=state.view(-1,groups,state.shape[-1])[:,0]; xi=xi.view(-1,groups,xi.shape[-1])[:,0]; act=act.view(-1,groups,act.shape[-1])[:,0]; f=f.view(-1,groups)[:,0]
if a_prev is not None: a_prev=a_prev.view(-1,groups,a_prev.shape[-1])[:,0]
n=min(a.limit,len(state)); state,xi,act,f=state[:n],xi[:n],act[:n],f[:n]
if a_prev is not None: a_prev=a_prev[:n]
st=ck['wm']; hidden=int(st['residual_net.0.weight'].shape[0]) if 'residual_net.0.weight' in st else 0; wm=WorldModel(residual_hidden=hidden).to(dev); wm.load_state_dict(st); wm.eval()
for p in wm.parameters(): p.requires_grad_(False)
action=act.clone().requires_grad_(True); pred=wm.step(unflatten_state(state),action,normalize_xi(xi));
# Legacy datasets lack a_prev; use current action only as a fallback.
from supermimic.kinematics import G1Kinematics
from supermimic.motion import MotionLib
from supermimic.mdp import MDP
from supermimic.robot import default_joint_pos_list
kin=G1Kinematics(dev); motion=MotionLib.from_dir(a.motion_dir or ck.get('args',{}).get('motion_dir','data/lafan1_g1'),kin,dev); mdp=MDP(kin,motion,torch.tensor(default_joint_pos_list(),device=dev))
r=mdp.reward(pred,f+1,action,action if a_prev is None else a_prev); grad=torch.autograd.grad(r.sum(),action)[0]; direction=grad/(grad.square().sum(-1,keepdim=True).sqrt().clamp_min(1e-6)); eps=a.epsilon*20
plus=(act+eps*direction).clamp(-20,20); minus=(act-eps*direction).clamp(-20,20)
# Candidate evaluator expects one row per state, with the base action paired.
payload={'candidate_frames':f.cpu(),'candidate_actions':plus.cpu(),'candidate_base_actions':act.cpu(),'gradient_minus_actions':minus.cpu(),'candidate_scores':r.detach().cpu(),'metadata':{'epsilon':a.epsilon,'samples':n}}
a.output.parent.mkdir(parents=True,exist_ok=True); torch.save(payload,a.output); print({'samples':n,'epsilon':a.epsilon,'grad_norm_mean':float(grad.norm(dim=-1).mean())})
