"""Offline audit of an ensemble residual gate with nominal fallback."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.buffer import ReplayBuffer
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, OBS_DIM, normalize_xi, unflatten_state
from supermimic.motion import MotionLib
from supermimic.networks import Policy, WorldModel
from supermimic.robot import default_joint_pos_list

def model_from_state(state, device):
    h = int(state["residual_net.0.weight"].shape[0]) if "residual_net.0.weight" in state else 0
    return WorldModel(residual_hidden=h).to(device)

def blend(res0, res1, nominal, gate):
    return tuple(torch.where(gate.view(-1, *([1] * (x.ndim - 1))), (x + y) * .5, n)
                 for x, y, n in zip(res0, res1, nominal))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("policy_checkpoint",type=Path); ap.add_argument("nominal",type=Path); ap.add_argument("member0",type=Path); ap.add_argument("member1",type=Path)
    ap.add_argument("--threshold",type=float,required=True); ap.add_argument("--horizon",type=int,default=16); ap.add_argument("--batch",type=int,default=128); ap.add_argument("--batches",type=int,default=8); ap.add_argument("--device",default="cuda:0"); args=ap.parse_args(); dev=args.device
    pck=torch.load(args.policy_checkpoint,map_location=dev,weights_only=False); a=pck.get("args",{}); state=pck["buffer"]; T,N=state["data"]["s"].shape[:2]; buf=ReplayBuffer(T,N,dev); buf.load_state_dict(state)
    kin=G1Kinematics(dev); motion=MotionLib.from_dir(a.get("motion_dir","data/lafan1_g1"),kin,dev); mdp=MDP(kin,motion,torch.tensor(default_joint_pos_list(),device=dev)); gf,hist=int(a.get("goal_frames",4)),int(a.get("hist",4)); pi=Policy(OBS_DIM,73*gf,hist=hist).to(dev); pi.load_state_dict(pck["pi"]); pi.eval()
    models=[]
    for path,use_res in ((args.nominal,False),(args.member0,True),(args.member1,True)):
        ck=torch.load(path,map_location=dev,weights_only=False); wm=model_from_state(ck["wm"],dev); wm.load_state_dict(ck["wm"]); wm.eval(); models.append(wm)
    rets=[]; gates=[]; diss=[]; all_dis=[]
    with torch.no_grad():
      for _ in range(args.batches):
        s,h=buf.sample_starts(args.batch,hist); S=unflatten_state(s["s"]); xi=s["xi"]; f=s["f"]; prev=s["a_prev"]; oh=mdp.obs(unflatten_state(h["s"]),h["a_prev"],h["xi"],h["f"]); goal=motion.goal(f,gf); total=torch.zeros(args.batch,device=dev); ng=0
        std=torch.sqrt(models[0].out_norm.var+models[0].out_norm.eps**2).clamp_min(1e-4)
        for t in range(args.horizon):
          act=pi(oh,goal); xn=normalize_xi(xi); sn=models[0].step(S,act,xn,use_residual=False); s0=models[1].step(S,act,xn,use_residual=True); s1=models[2].step(S,act,xn,use_residual=True)
          d=(models[0].state_error(s0,s1)/std).square().mean(-1); all_dis.append(d); g=d<=args.threshold; ng+=g.float();
          chosen=blend(s0,s1,sn,g)
          total+=mdp.reward(chosen,f,act,prev); obs=mdp.obs(chosen,act,xi,f+1); oh=torch.cat((oh[:,1:],obs.unsqueeze(1)),1); goal=motion.goal(f+1,gf); prev=act; f=f+1; S=chosen
        rets.append(total/args.horizon); gates.append(ng/args.horizon); diss.append(d)
    r=torch.cat(rets); g=torch.cat(gates); d=torch.cat(diss); ad=torch.cat(all_dis)
    print(json.dumps({"threshold":args.threshold,"horizon":args.horizon,"samples":int(r.numel()),"gated_fraction":float(g.mean()),"imagined_return_mean":float(r.mean()),"imagined_return_p10":float(torch.quantile(r,.1)),"final_disagreement_mean":float(d.mean()),"rollout_disagreement_quantiles":{"p10":float(torch.quantile(ad,.1)),"p25":float(torch.quantile(ad,.25)),"p50":float(torch.quantile(ad,.5)),"p75":float(torch.quantile(ad,.75)),"p90":float(torch.quantile(ad,.9))}},indent=2))
if __name__=="__main__": main()
