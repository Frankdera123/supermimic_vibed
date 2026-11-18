"""Audit gradient scale/alignment of teacher-forced and free-running WM losses."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.buffer import ReplayBuffer
from supermimic.mdp import normalize_xi, unflatten_state
from supermimic.networks import WorldModel

def model_from_state(state, device):
    hidden = int(state["residual_net.0.weight"].shape[0]) if "residual_net.0.weight" in state else 0
    return WorldModel(residual_hidden=hidden).to(device)

def pred_teacher(wm, w):
    out=[]
    for j in range(w["s"].shape[1]):
        out.append(wm.step(unflatten_state(w["s"][:,j]), w["a"][:,j], normalize_xi(w["xi"][:,j])))
    return tuple(torch.stack([p[i] for p in out],1) for i in range(6))

def pred_free(wm, w):
    st=unflatten_state(w["s"][:,0]); out=[]
    for j in range(w["s"].shape[1]):
        st=wm.step(st,w["a"][:,j],normalize_xi(w["xi"][:,j])); out.append(st)
    return tuple(torch.stack([p[i] for p in out],1) for i in range(6))

def loss(wm, pred, w):
    std=torch.sqrt(wm.out_norm.var+wm.out_norm.eps**2).clamp_min(1e-4)
    e=(wm.state_error(pred,unflatten_state(w["s1"])) / std).square().mean(-1)
    m=w["mask"]
    return (e*m).sum()/m.sum().clamp_min(1.)

def flat(grads):
    return torch.cat([g.detach().flatten() for g in grads if g is not None])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("checkpoint",type=Path); ap.add_argument("--main_window",type=int,default=4); ap.add_argument("--aux_window",type=int,default=8); ap.add_argument("--batch",type=int,default=32); ap.add_argument("--batches",type=int,default=16); ap.add_argument("--device",default="cuda:0"); args=ap.parse_args()
    ck=torch.load(args.checkpoint,map_location=args.device,weights_only=False); st=ck.get("buffer")
    if st is None: ap.error("checkpoint must contain replay buffer state")
    T,N=st["data"]["s"].shape[:2]; buf=ReplayBuffer(T,N,args.device); buf.load_state_dict(st)
    wm=model_from_state(ck["wm"],args.device); wm.load_state_dict(ck["wm"]); wm.train()
    params=[p for p in wm.parameters() if p.requires_grad]
    rows=[]
    for _ in range(args.batches):
        wm.zero_grad(set_to_none=True)
        w1=buf.sample_windows(args.batch,args.main_window)
        l1=loss(wm,pred_teacher(wm,w1),w1)
        g1=flat(torch.autograd.grad(l1,params,retain_graph=False,allow_unused=True))
        wm.zero_grad(set_to_none=True)
        w2=buf.sample_windows(args.batch,args.aux_window)
        l2=loss(wm,pred_free(wm,w2),w2)
        g2=flat(torch.autograd.grad(l2,params,retain_graph=False,allow_unused=True))
        cos=torch.nn.functional.cosine_similarity(g1,g2,dim=0)
        rows.append((l1.detach(),l2.detach(),g1.norm(),g2.norm(),cos))
    x=torch.stack([torch.stack(r) for r in rows]).mean(0)
    print(json.dumps({"checkpoint":str(args.checkpoint),"main_window":args.main_window,"aux_window":args.aux_window,"batches":args.batches,"main_loss":float(x[0]),"aux_loss":float(x[1]),"main_grad_norm":float(x[2]),"aux_grad_norm":float(x[3]),"aux_to_main_grad_ratio":float(x[3]/x[2].clamp_min(1e-12)),"gradient_cosine":float(x[4])},indent=2))
if __name__=="__main__": main()
