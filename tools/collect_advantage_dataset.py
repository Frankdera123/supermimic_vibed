"""Collect same-state candidate actions labeled by short real-simulator return advantage."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import add_launcher_args, launch_simulation
from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, OBS_DIM, flatten_state
from supermimic.motion import MotionLib
from supermimic.networks import Policy
from supermimic.robot import default_joint_pos_list
from supermimic.strategic_critic import StrategicErrorCritic

ap=argparse.ArgumentParser()
ap.add_argument('checkpoint',type=Path); ap.add_argument('--output',type=Path,required=True)
ap.add_argument('--num_states',type=int,default=32); ap.add_argument('--steps',type=int,default=20)
ap.add_argument('--horizon',type=int,default=4); ap.add_argument('--action_candidates',type=int,default=4)
ap.add_argument('--candidate_noise',type=float,default=.15); ap.add_argument('--motion_dir',default=None); ap.add_argument('--seed',type=int,default=0)
ap.add_argument('--critic_checkpoint',type=Path,default=None)
add_launcher_args(ap); args=ap.parse_args(); dev=args.device
if min(args.num_states,args.steps,args.horizon,args.action_candidates)<1 or args.candidate_noise<0: ap.error('invalid collection arguments')
torch.manual_seed(args.seed)
ck=torch.load(args.checkpoint,map_location=dev,weights_only=False); base=ck.get('args',{})
motion_dir=args.motion_dir or base.get('motion_dir'); kin=G1Kinematics(dev); motion=MotionLib.from_dir(motion_dir,kin,dev)
mdp=MDP(kin,motion,torch.tensor(default_joint_pos_list(),device=dev)); goal_frames=int(base.get('goal_frames',10)); hist=int(base.get('hist',10))
policy=Policy(OBS_DIM,73*goal_frames,hist=hist).to(dev); policy.load_state_dict(ck['pi']); policy.eval()
adv_critic=None
if args.critic_checkpoint is not None:
 cck=torch.load(args.critic_checkpoint,map_location=dev,weights_only=False)
 adv_critic=StrategicErrorCritic(hidden=int(cck['hidden'])).to(dev); adv_critic.load_state_dict(cck['model']); adv_critic.eval()
 cmean=cck['mean'].to(dev); cstd=cck['std'].to(dev); cym=cck['target_mean'].to(dev); cys=cck['target_std'].to(dev)
g=args.action_candidates; n=args.num_states; total=n*g
sim_cfg=make_sim(dev,SimpleNamespace(num_envs=total))
all_s=[]; all_a=[]; all_xi=[]; all_aprev=[]; all_f=[]; all_ret=[]; all_selected=[]
with launch_simulation(sim_cfg,args):
 env=TrackingSim(sim_cfg,total,motion,mdp,gamma=float(base.get('gamma',.99)),eval_mode=True,reset_noise=False)
 for _ in range(args.steps):
  frames,_=env.sample_frames(n); frames=frames.repeat_interleave(g); env.reset(torch.arange(total,device=dev),frames); env.refresh()
  S0=env.state(); xi0=env.xi.clone(); aprev0=env.a_prev.clone(); f0=env.f.clone(); obs_hist=torch.zeros(total,hist,OBS_DIM,device=dev)
  o=mdp.obs(S0,env.a_prev,env.xi,env.f); obs_hist[:]=o[:,None]
  base_a=policy(obs_hist,motion.goal(env.f,goal_frames)).clamp(-20,20).detach().view(n,g,-1)[:,0]
  noise=torch.randn(n,g,base_a.shape[-1],device=dev)*args.candidate_noise*20; noise[:,0].zero_()
  actions=(base_a[:,None]+noise).clamp(-20,20).reshape(total,-1)
  selected_idx=torch.zeros(n,dtype=torch.long,device=dev)
  if adv_critic is not None:
   with torch.no_grad():
    fs=flatten_state(S0).view(n,g,-1).reshape(total,-1); fx=xi0.view(n,g,-1)[:,0][:,None].expand(-1,g,-1).reshape(total,-1)
    if fs.shape[1] != 71 or actions.shape[1] != 29 or fx.shape[1] != 33:
     raise RuntimeError(f'candidate feature shapes state={tuple(fs.shape)} action={tuple(actions.shape)} xi={tuple(fx.shape)}')
    scores=(adv_critic((adv_critic.features(fs,actions,fx)-cmean)/cstd)*cys+cym).view(n,g)
    selected_idx=scores.argmax(1)
    selected_idx=torch.where(scores.max(1).values>0,selected_idx,torch.zeros_like(selected_idx))
  ret=torch.zeros(total,device=dev); alive=torch.ones(total,dtype=torch.bool,device=dev)
  for t in range(args.horizon):
   if t>0:
    S=env.state(); o=mdp.obs(S,env.a_prev,env.xi,env.f); obs_hist=torch.cat((obs_hist[:,1:],o[:,None]),1)
    actions=policy(obs_hist,motion.goal(env.f,goal_frames)).clamp(-20,20).detach()
   actions=torch.where(alive[:,None],actions,torch.zeros_like(actions)); out=env.step(actions); ret += out['r']*alive.float(); alive &= ~out['done']
  all_s.append(flatten_state(S0).detach().cpu()); all_a.append((base_a[:,None]+noise).reshape(total,-1).detach().cpu()); all_xi.append(xi0.detach().cpu()); all_aprev.append(aprev0.detach().cpu()); all_f.append(f0.detach().cpu()); all_ret.append(ret.detach().cpu())
  all_selected.append(selected_idx.detach().cpu())
payload={'s':torch.cat(all_s),'a':torch.cat(all_a),'xi':torch.cat(all_xi),'a_prev':torch.cat(all_aprev),'f':torch.cat(all_f),'return':torch.cat(all_ret)}
r=payload['return'].view(-1,g); payload['advantage']=(r-r[:,0:1]).reshape(-1); payload['metadata']={'checkpoint':str(args.checkpoint),'num_states':n,'steps':args.steps,'horizon':args.horizon,'action_candidates':g,'candidate_noise':args.candidate_noise,'seed':args.seed}
if adv_critic is not None:
 selected_idx_all=torch.cat(all_selected)
 selected=r.gather(1,selected_idx_all.view(-1,1)).squeeze(1)
 payload['metadata'].update({'critic_checkpoint':str(args.critic_checkpoint),'selected_return_mean':float(selected.mean()),'selected_advantage_mean':float((selected-r[:,0]).mean()),'selected_positive_fraction':float((selected>r[:,0]).float().mean()),'base_return_mean':float(r[:,0].mean())})
args.output.parent.mkdir(parents=True,exist_ok=True); torch.save(payload,args.output)
print({'rows':len(payload['s']),'return_mean':float(r.mean()),'base_return_mean':float(r[:,0].mean()),'best_advantage_mean':float(r.max(1).values.mean()),'positive_candidate_fraction':float((payload['advantage']>0).float().mean())})
