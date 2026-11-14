import time
import pandas as pd
from src.load import time_split
from src.detectors import fan

print("Loading from cache.pkl...")
t0 = time.time()
df = pd.read_pickle("data/cache.pkl")
print(f"Loaded in {time.time()-t0:.2f}s")

train, val, test = time_split(df)
val_t_min = val["t"].min()
val_24h = val[(val["t"] >= val_t_min) & (val["t"] <= val_t_min + 24.0)].copy().reset_index(drop=True)
n_val_24h = len(val_24h)
n_illicit = int(val_24h["label"].sum())
print(f"Val 24h window: {n_val_24h:,} rows | Illicit: {n_illicit} | t: [{val_24h['t'].min():.2f}h, {val_24h['t'].max():.2f}h]")

k_vals = [12, 20, 30]
w_vals = [12.0, 48.0]

print("\n=== TUNING FAN-OUT (key=src, other=dst) ===")
fo_results = []
for k in k_vals:
    for w in w_vals:
        t0 = time.time()
        hits = fan(val_24h, key="src", other="dst", k=k, W=w)
        dt = time.time() - t0
        eids = set().union(*(h["eids"] for h in hits)) if hits else set()
        illicit_cov = int(val_24h[val_24h["eid"].isin(eids)]["label"].sum()) if eids else 0
        prec = illicit_cov / len(eids) if eids else 0.0
        fo_results.append({"k": k, "W": w, "hits": len(hits), "covered_tx": len(eids), "illicit_covered": illicit_cov, "precision": prec, "time_s": dt})
        print(f"  k={k:>2}, W={w:>2.0f}h -> hits={len(hits):>5}, covered_tx={len(eids):>6}, illicit_covered={illicit_cov:>3}, precision={prec:.6f}, time={dt:.2f}s")

print("\n=== TUNING FAN-IN (key=dst, other=src) ===")
fi_results = []
for k in k_vals:
    for w in w_vals:
        t0 = time.time()
        hits = fan(val_24h, key="dst", other="src", k=k, W=w)
        dt = time.time() - t0
        eids = set().union(*(h["eids"] for h in hits)) if hits else set()
        illicit_cov = int(val_24h[val_24h["eid"].isin(eids)]["label"].sum()) if eids else 0
        prec = illicit_cov / len(eids) if eids else 0.0
        fi_results.append({"k": k, "W": w, "hits": len(hits), "covered_tx": len(eids), "illicit_covered": illicit_cov, "precision": prec, "time_s": dt})
        print(f"  k={k:>2}, W={w:>2.0f}h -> hits={len(hits):>5}, covered_tx={len(eids):>6}, illicit_covered={illicit_cov:>3}, precision={prec:.6f}, time={dt:.2f}s")
