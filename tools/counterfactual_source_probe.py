"""Measure whether counterfactual feature families reveal dataset/source identity."""
from __future__ import annotations
import argparse
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

ap = argparse.ArgumentParser()
ap.add_argument("datasets", nargs="+", type=str)
args = ap.parse_args()
if len(args.datasets) < 2:
    ap.error("provide at least two datasets")
families = {"xi": [], "state": [], "response": []}
labels = []
for label, path in enumerate(args.datasets):
    d = torch.load(path, map_location="cpu", weights_only=False)
    s0 = d["pairs"][0]["s0"].float()
    response = torch.cat([
        (p["plus"].float() - p["minus"].float()).reshape(s0.shape[0], -1)
        for p in d["pairs"]], 1)
    response = response / response.norm(dim=1, keepdim=True).clamp_min(1e-6)
    families["xi"].append(d["xi"].float().numpy())
    families["state"].append(s0.numpy())
    families["response"].append(response.numpy())
    labels.extend([label] * s0.shape[0])
y = np.asarray(labels)
cv = StratifiedKFold(5, shuffle=True, random_state=23)
out = {}
for name, blocks in families.items():
    X = np.concatenate(blocks, 0)
    clf = LogisticRegression(max_iter=3000)
    if len(np.unique(y)) == 2:
        scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
        out[name] = {"fold_auc": [float(x) for x in scores], "mean_auc": float(scores.mean())}
    else:
        # Multiclass probabilities are unstable for these high-dimensional
        # response vectors; report all pairwise source probes instead.
        pairs = {}
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                Xp = np.concatenate((blocks[i], blocks[j]), 0)
                yp = np.concatenate((np.zeros(len(blocks[i])), np.ones(len(blocks[j]))))
                scores = cross_val_score(LogisticRegression(max_iter=3000), Xp, yp,
                                         cv=StratifiedKFold(5, shuffle=True, random_state=23),
                                         scoring="roc_auc")
                pairs[f"{i}_vs_{j}"] = {"fold_auc": [float(x) for x in scores],
                                        "mean_auc": float(scores.mean())}
        out[name] = {"pairwise": pairs}
print(out)
