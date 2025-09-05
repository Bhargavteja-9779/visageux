from pathlib import Path
import numpy as np, pandas as pd, torch, json
from .utils import load_sources, FEATURES, zscore_apply, build_sequences

REPO = Path(__file__).resolve().parents[2]
CHK = REPO / "models" / "checkpoints" / "dropper_gru.pt"
OUT = REPO / "data" / "predictions" / "drop_prob_5s.parquet"

class GRUDrop(torch.nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.gru = torch.nn.GRU(input_size=in_dim, hidden_size=hidden, num_layers=1, batch_first=True)
        self.head = torch.nn.Sequential(torch.nn.Linear(hidden,32), torch.nn.ReLU(), torch.nn.Linear(32,1))
    def forward(self,x):
        o,_ = self.gru(x); return self.head(o[:,-1,:]).squeeze(-1)

def main():
    ck = torch.load(CHK, map_location="cpu")
    cfg = ck["config"]; stats = ck["stats"]
    SEQLEN = int(cfg["SEQLEN"]); FEAT_LIST = cfg["FEATURES"]

    ev, w = load_sources()
    Xdf = w[["sess_key"]+FEAT_LIST].copy()
    X = zscore_apply(Xdf[FEAT_LIST], stats)
    idx_seq, _ = build_sequences(w, pd.Series(np.zeros(len(w),dtype=int), index=w.index), L=SEQLEN)
    if idx_seq.size == 0:
        print("[infer] No sequences to score."); return
    X_seq = np.stack([X[idxs] for idxs in idx_seq], axis=0)

    # baseline?
    if ck.get("state_dict") is None:
        prior = float(cfg.get("baseline_prior", 0.5))
        last_idxs = idx_seq[:, -1]
        out = w.loc[last_idxs, ["sess_key","w_start","w_end"]].copy()
        out["p_drop_next_10s"] = prior
        OUT.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(OUT, index=False)
        print(f"[infer] baseline prior={prior:.3f} → {OUT} ({len(out)} rows)")
        return

    model = GRUDrop(in_dim=len(FEAT_LIST), hidden=int(cfg["HIDDEN"]))
    model.load_state_dict(ck["state_dict"]); model.eval()

    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(X_seq))).cpu().numpy()

    last_idxs = idx_seq[:, -1]
    out = w.loc[last_idxs, ["sess_key","w_start","w_end"]].copy()
    out["p_drop_next_10s"] = probs
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"[infer] wrote {len(out)} rows → {OUT}")

if __name__=="__main__": main()
