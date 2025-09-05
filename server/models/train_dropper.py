from pathlib import Path
import numpy as np, pandas as pd, json, torch
import torch.nn as nn, torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from .utils import load_sources, compute_next_event_gap, make_labels, FEATURES, zscore_fit, zscore_apply, build_sequences

REPO = Path(__file__).resolve().parents[2]
CHKDIR = REPO / "models" / "checkpoints"; CHKDIR.mkdir(parents=True, exist_ok=True)

SEQLEN_TARGET = 6
FEAT_DIM = len(FEATURES); HIDDEN = 48; EPOCHS=8; BS=128; LR=1e-3; HORIZON=10.0

class GRUDrop(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden,32), nn.ReLU(), nn.Linear(32,1))
    def forward(self,x):
        o,_ = self.gru(x); return self.head(o[:,-1,:]).squeeze(-1)

def main():
    ev, w = load_sources()
    if len(w)==0:
        print("[train] No windows. Build features first."); return

    gaps = compute_next_event_gap(ev, w)
    y = make_labels(w, gaps, horizon_sec=HORIZON)
    Xdf = w[["sess_key"]+FEATURES].copy()
    stats = zscore_fit(Xdf[FEATURES]); X = zscore_apply(Xdf[FEATURES], stats)

    # choose L
    per = w.groupby("sess_key")["w_start"].size().values
    max_win = int(per.max()) if len(per) else 0
    if max_win < 2: print("[train] Too few windows per session."); return
    L = min(SEQLEN_TARGET, max_win)
    print(f"[train] Using sequence length L={L}")

    idx_seq, y_seq = build_sequences(w, y, L=L)
    if idx_seq.size == 0:
        print("[train] No sequences. Seed more events."); return
    X_seq = np.stack([X[idxs] for idxs in idx_seq], axis=0)

    # single-class guard
    if len(np.unique(y_seq)) < 2:
        print("[train] Labels are single-class; training a bias-only baseline.")
        # Save a tiny baseline checkpoint with constant logit
        prior = float(np.mean(y_seq))
        ck = {"state_dict": None, "stats": stats,
              "config":{"SEQLEN":int(L),"FEATURES":FEATURES,"HORIZON":HORIZON,
                        "FEAT_DIM":FEAT_DIM,"HIDDEN":HIDDEN, "baseline_prior": prior}}
        torch.save(ck, CHKDIR/"dropper_gru.pt")
        with open(CHKDIR/"dropper_gru.meta.json","w") as f: json.dump({"best_val_auroc": None, "note":"baseline"}, f)
        print(f"  ↳ saved baseline prior={prior:.3f}")
        return

    Xtr, Xte, ytr, yte = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq)
    device = torch.device("cpu")
    model = GRUDrop(FEAT_DIM, HIDDEN).to(device)
    opt = optim.Adam(model.parameters(), lr=LR)
    loss_bce = nn.BCEWithLogitsLoss()

    def run_epoch(Xb, yb, train=True):
        model.train(train); losses=[]
        for i in range(0,len(Xb),BS):
            xb = torch.from_numpy(Xb[i:i+BS]).to(device)
            yb_ = torch.from_numpy(yb[i:i+BS]).float().to(device)
            with torch.set_grad_enabled(train):
                lo = model(xb); loss = loss_bce(lo, yb_)
                if train: opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        return float(np.mean(losses))

    best_auc = 0.0
    for ep in range(1,EPOCHS+1):
        tl = run_epoch(Xtr,ytr,True); vl = run_epoch(Xte,yte,False)
        with torch.no_grad():
            pr = torch.sigmoid(model(torch.from_numpy(Xte).to(device))).cpu().numpy()
            auc = roc_auc_score(yte, pr) if len(np.unique(yte))>1 else np.nan
            ap  = average_precision_score(yte, pr) if len(np.unique(yte))>1 else np.nan
        print(f"[epoch {ep}/{EPOCHS}] train {tl:.4f} | val {vl:.4f} | AUROC {auc:.3f} | AUPRC {ap:.3f}")
        if np.isfinite(auc) and auc>best_auc:
            best_auc=auc
            ck={"state_dict":model.state_dict(),"stats":stats,
                "config":{"SEQLEN":int(L),"FEATURES":FEATURES,"HORIZON":HORIZON,
                          "FEAT_DIM":FEAT_DIM,"HIDDEN":HIDDEN}}
            torch.save(ck, CHKDIR/"dropper_gru.pt")
            with open(CHKDIR/"dropper_gru.meta.json","w") as f: json.dump({"best_val_auroc": float(auc)}, f)
            print(f"  ↳ saved checkpoint AUROC={auc:.3f}")
if __name__=="__main__": main()
