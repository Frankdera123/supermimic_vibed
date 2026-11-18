"""Compare a fixed policy's imagined returns under nominal and bootstrap WMs."""
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
    key = "residual_net.0.weight"
    hidden = int(state[key].shape[0]) if key in state else 0
    return WorldModel(residual_hidden=hidden).to(device)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy_checkpoint", type=Path)
    ap.add_argument("model_checkpoints", nargs="+", type=Path)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    pck = torch.load(args.policy_checkpoint, map_location=args.device, weights_only=False)
    state = pck.get("buffer")
    if state is None: ap.error("policy checkpoint must contain replay buffer")
    a = pck.get("args", {}); dev = args.device
    goal_frames, hist = int(a.get("goal_frames", 4)), int(a.get("hist", 4))
    T, N = state["data"]["s"].shape[:2]
    buf = ReplayBuffer(T, N, dev); buf.load_state_dict(state)
    kin = G1Kinematics(dev); motion = MotionLib.from_dir(a.get("motion_dir", "data/lafan1_g1"), kin, dev)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
    pi = Policy(OBS_DIM, 73 * goal_frames, hist=hist).to(dev); pi.load_state_dict(pck["pi"]); pi.eval()
    models=[]
    for path in args.model_checkpoints:
        ck=torch.load(path,map_location=dev,weights_only=False); wm=model_from_state(ck["wm"],dev); wm.load_state_dict(ck["wm"]); wm.eval(); models.append((path.name,wm,"residual_net.0.weight" in ck["wm"]))
    returns=[[] for _ in models]; disagreements=[]
    with torch.no_grad():
        for _ in range(args.batches):
            s,h=buf.sample_starts(args.batch,hist)
            S0=unflatten_state(s["s"]); xi=s["xi"]; f=s["f"]
            obs_hist=mdp.obs(unflatten_state(h["s"]),h["a_prev"],h["xi"],h["f"])
            goal=motion.goal(f,goal_frames); a_prev=s["a_prev"]
            states=[tuple(x.clone() for x in S0) for _ in models]
            total=[torch.zeros(args.batch,device=dev) for _ in models]
            final_dis=None
            for t in range(args.horizon):
                act=pi(obs_hist,goal)
                xi_n=normalize_xi(xi)
                next_states=[]
                for j,(_,wm,use_res) in enumerate(models):
                    ns=wm.step(states[j],act,xi_n,use_residual=use_res)
                    total[j]+=mdp.reward(ns,f,act,a_prev)
                    next_states.append(ns)
                # Pairwise normalized state disagreement at this imagined step.
                if len(next_states)>=2:
                    std=torch.sqrt(models[0][1].out_norm.var+models[0][1].out_norm.eps**2).clamp_min(1e-4)
                    ds=[]
                    for j in range(1,len(next_states)):
                        ds.append((models[0][1].state_error(next_states[0],next_states[j])/std).square().mean(-1))
                    final_dis=torch.stack(ds).mean(0)
                # Policy actions are generated from the first model's state.
                obs=mdp.obs(next_states[0],act,xi,f+1)
                obs_hist=torch.cat((obs_hist[:,1:],obs.unsqueeze(1)),1)
                goal=motion.goal(f+1,goal_frames); a_prev=act; f=f+1; states=next_states
            for j in range(len(models)): returns[j].append(total[j]/args.horizon)
            if final_dis is not None: disagreements.append(final_dis)
    out={"policy_checkpoint":str(args.policy_checkpoint),"models":[x[0] for x in models],"horizon":args.horizon,"samples":args.batch*args.batches}
    out["imagined_return"]=[{"model":models[j][0],"mean":float(torch.cat(returns[j]).mean()),"p10":float(torch.quantile(torch.cat(returns[j]),.1)),"p50":float(torch.quantile(torch.cat(returns[j]),.5)),"p90":float(torch.quantile(torch.cat(returns[j]),.9))} for j in range(len(models))]
    if disagreements:
        d=torch.cat(disagreements); out["final_disagreement"]={"mean":float(d.mean()),"p50":float(torch.quantile(d,.5)),"p90":float(torch.quantile(d,.9))}
    print(json.dumps(out,indent=2))
if __name__=="__main__": main()
