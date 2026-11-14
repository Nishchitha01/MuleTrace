"""
run_all.py — MuleTrace evaluation pipeline.

Usage:
    python -u run_all.py --split val
    python -u run_all.py --split test

Features:
  - Caching with data/cache.pkl.
  - Edge-level Union-Find clustering with max-300 account split limit.
  - Tuned fan detectors (fan-out k=30, W=12h; fan-in k=12, W=12h).
  - Transaction-level PRF based strictly on alert edge eids.
  - Matched-budget baseline comparison (top N by amount_usd).
  - Ranked-alert metrics for top 50, 100, 250, 500 alerts.
  - Strict and relaxed ring recall with per-ring coverage fractions.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Ensure project root is on sys.path ─────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.load import load, time_split
from src.patterns_fixed import parse_patterns
from src.baseline import tune_baseline, evaluate_baseline, baseline
from src.evaluate import prf, account_level_prf, ring_recall, print_table
from src.detectors import fan, chains, temporal_cycles, scatter_gather, gather_scatter
from src.cluster import merge_hits
from src.score import clusters_to_alerts, save_alerts


def log(msg: str) -> None:
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now_str}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Transaction & Account Flagging Definitions
# ---------------------------------------------------------------------------

def flagged_eids_from_alerts(alerts: list[dict]) -> set[int]:
    """
    A transaction is flagged at transaction level ONLY if its eid appears
    directly in an alert's edges list.
    """
    flagged: set[int] = set()
    for a in alerts:
        for e in a.get("edges", []):
            if "eid" in e:
                flagged.add(e["eid"])
    return flagged


def flagged_accounts_from_alerts(alerts: list[dict]) -> set[str]:
    """
    An account is flagged at account level if it appears in any alert's accounts set.
    """
    accounts: set[str] = set()
    for a in alerts:
        accounts.update(a.get("accounts", []))
    return accounts


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MuleTrace Pipeline Runner")
    parser.add_argument("--split", choices=["val", "test"], default="val",
                        help="Split to evaluate: 'val' or 'test' (default: val)")
    parser.add_argument("--enable-composites", action="store_true", default=False,
                        help="Enable scatter_gather and gather_scatter (default: False)")
    parser.add_argument("--slice", type=int, default=None,
                        help="Optional slice limit for quick testing (default: None for full data)")
    args = parser.parse_args()

    eval_split_name = args.split
    enable_composites = args.enable_composites
    slice_n = args.slice

    t_start_total = time.time()
    log(f"============================================================")
    log(f"  MuleTrace Pipeline — Evaluating split: {eval_split_name.upper()}")
    log(f"  Composites: {'ENABLED' if enable_composites else 'DISABLED (takes >2m on 1-day window)'}")
    log(f"  Data slice: {f'{slice_n:,} rows' if slice_n else 'FULL DATASET'}")
    log(f"============================================================")

    # 1. LOAD & TIME SPLIT (WITH CACHING)
    cache_path = ROOT / "data" / "cache.pkl"
    if cache_path.exists() and slice_n is None:
        log(f"Stage 1: Loading full dataset from cache ({cache_path})...")
        t0 = time.time()
        df = pd.read_pickle(str(cache_path))
        log(f"Dataset loaded from cache in {time.time() - t0:.2f}s | Rows: {len(df):,} | Illicit: {int(df['label'].sum()):,}")
    else:
        log("Stage 1: Loading dataset from CSV...")
        data_path = str(ROOT / "data" / "HI-Small_Trans.csv")
        t0 = time.time()
        df = load(data_path, nrows=slice_n)
        log(f"Dataset loaded in {time.time() - t0:.2f}s | Rows: {len(df):,} | Illicit: {int(df['label'].sum()):,}")
        if slice_n is None:
            log("Caching dataset to data/cache.pkl...")
            df.to_pickle(str(cache_path))
            log("Dataset successfully cached.")

    log("Stage 1b: Splitting dataset (60/20/20 time split)...")
    train, val, test = time_split(df)
    log(f"  Train : {len(train):,} rows | Illicit: {int(train['label'].sum()):,} | t: [{train['t'].min():.2f}h, {train['t'].max():.2f}h]")
    log(f"  Val   : {len(val):,} rows | Illicit: {int(val['label'].sum()):,} | t: [{val['t'].min():.2f}h, {val['t'].max():.2f}h]")
    log(f"  Test  : {len(test):,} rows | Illicit: {int(test['label'].sum()):,} | t: [{test['t'].min():.2f}h, {test['t'].max():.2f}h]")

    eval_df = val if eval_split_name == "val" else test
    log(f"Active evaluation split: {eval_split_name.upper()} ({len(eval_df):,} rows)")

    # 2. BASELINE TUNING & EVALUATION
    log("Stage 2: Tuning baseline on FULL train split...")
    amt_grid = [1000.0, 2500.0, 5000.0, 10000.0, 25000.0, 50000.0, 100000.0, 250000.0, 500000.0, 1000000.0]
    n_grid = [3, 5, 10, 20, 50, 100, 200]
    sum_grid = [5000.0, 10000.0, 25000.0, 50000.0, 100000.0, 250000.0, 500000.0, 1000000.0, 2500000.0, 5000000.0]

    t0 = time.time()
    best_params = tune_baseline(train, amt_grid=amt_grid, n_grid=n_grid, sum_grid=sum_grid)
    log(f"Baseline tuned in {time.time() - t0:.2f}s.")
    log(f"  Best params picked on Train: amt_thr=${best_params['amt_thr']:,.2f}, daily_n={best_params['daily_n']}, daily_sum=${best_params['daily_sum']:,.2f}")

    # Evaluate Tuned Baseline on Train and Eval split
    bl_train = evaluate_baseline(train, best_params)
    bl_eval = evaluate_baseline(eval_df, best_params)

    # Evaluate Plain $10,000 rule on Train and Eval split
    plain_10k_params = {"amt_thr": 10000.0, "daily_n": float("inf"), "daily_sum": float("inf")}
    plain_train = evaluate_baseline(train, plain_10k_params)
    plain_eval = evaluate_baseline(eval_df, plain_10k_params)

    log(f"Baseline Comparison (Train split):")
    log(f"  Tuned Baseline -> P={bl_train['precision']:.4f}, R={bl_train['recall']:.4f}, F1={bl_train['f1']:.4f} (TP={bl_train['tp']}, FP={bl_train['fp']})")
    log(f"  Plain $10k     -> P={plain_train['precision']:.4f}, R={plain_train['recall']:.4f}, F1={plain_train['f1']:.4f} (TP={plain_train['tp']}, FP={plain_train['fp']})")

    log(f"Baseline Comparison ({eval_split_name.upper()} split):")
    log(f"  Tuned Baseline -> P={bl_eval['precision']:.4f}, R={bl_eval['recall']:.4f}, F1={bl_eval['f1']:.4f} (TP={bl_eval['tp']}, FP={bl_eval['fp']})")
    log(f"  Plain $10k     -> P={plain_eval['precision']:.4f}, R={plain_eval['recall']:.4f}, F1={plain_eval['f1']:.4f} (TP={plain_eval['tp']}, FP={plain_eval['fp']})")

    # 3. RUN TUNED DETECTORS ON EVAL SPLIT
    # Fan parameters chosen from tuning on first 24h of val:
    # fan-out: k=30, W=12.0h (highest precision among >= 30 illicit covered)
    # fan-in:  k=12, W=12.0h (highest coverage & precision)
    FO_K, FO_W = 30, 12.0
    FI_K, FI_W = 12, 12.0

    log(f"Stage 3: Running detectors on {eval_split_name.upper()} split...")
    detector_stats = {}

    # fan-out
    log(f"  Running fan-out detector (k={FO_K}, W={FO_W}h)...")
    t0 = time.time()
    fo_hits = fan(eval_df, key="src", other="dst", k=FO_K, W=FO_W)
    dt_fo = time.time() - t0
    fo_eids = set().union(*(h["eids"] for h in fo_hits)) if fo_hits else set()
    fo_prec = float(eval_df[eval_df["eid"].isin(fo_eids)]["label"].mean()) if fo_eids else 0.0
    detector_stats["fan_out"] = {
        "hits": len(fo_hits),
        "covered_transactions": len(fo_eids),
        "precision": fo_prec,
        "time_s": dt_fo,
    }
    log(f"    fan-out finished in {dt_fo:.2f}s | Hits: {len(fo_hits):,} | Covered txs: {len(fo_eids):,} | Precision: {fo_prec:.4f}")

    # fan-in
    log(f"  Running fan-in detector (k={FI_K}, W={FI_W}h)...")
    t0 = time.time()
    fi_hits = fan(eval_df, key="dst", other="src", k=FI_K, W=FI_W)
    dt_fi = time.time() - t0
    fi_eids = set().union(*(h["eids"] for h in fi_hits)) if fi_hits else set()
    fi_prec = float(eval_df[eval_df["eid"].isin(fi_eids)]["label"].mean()) if fi_eids else 0.0
    detector_stats["fan_in"] = {
        "hits": len(fi_hits),
        "covered_transactions": len(fi_eids),
        "precision": fi_prec,
        "time_s": dt_fi,
    }
    log(f"    fan-in finished in {dt_fi:.2f}s | Hits: {len(fi_hits):,} | Covered txs: {len(fi_eids):,} | Precision: {fi_prec:.4f}")

    # chains
    log("  Running chains detector (D=6h, keep=0.70)...")
    t0 = time.time()
    ch_hits = chains(eval_df, D=6.0, keep=0.70)
    dt_ch = time.time() - t0
    ch_eids = set().union(*(h["eids"] for h in ch_hits)) if ch_hits else set()
    ch_prec = float(eval_df[eval_df["eid"].isin(ch_eids)]["label"].mean()) if ch_eids else 0.0
    detector_stats["chain"] = {
        "hits": len(ch_hits),
        "covered_transactions": len(ch_eids),
        "precision": ch_prec,
        "time_s": dt_ch,
    }
    log(f"    chains finished in {dt_ch:.2f}s | Hits: {len(ch_hits):,} | Covered txs: {len(ch_eids):,} | Precision: {ch_prec:.4f}")

    # temporal_cycles
    log("  Running temporal_cycles detector (max_len=5, W=168h)...")
    t0 = time.time()
    cy_hits = temporal_cycles(eval_df, max_len=5, W=168.0)
    dt_cy = time.time() - t0
    cy_eids = set().union(*(h["eids"] for h in cy_hits)) if cy_hits else set()
    cy_prec = float(eval_df[eval_df["eid"].isin(cy_eids)]["label"].mean()) if cy_eids else 0.0
    detector_stats["cycle"] = {
        "hits": len(cy_hits),
        "covered_transactions": len(cy_eids),
        "precision": cy_prec,
        "time_s": dt_cy,
    }
    log(f"    temporal_cycles finished in {dt_cy:.2f}s | Hits: {len(cy_hits):,} | Covered txs: {len(cy_eids):,} | Precision: {cy_prec:.4f}")

    # Optional composites
    all_hits = fo_hits + fi_hits + ch_hits + cy_hits
    if enable_composites:
        log("  Running composites (scatter_gather & gather_scatter)...")
        sg_hits = scatter_gather(eval_df, W=48.0, k=8)
        gs_hits = gather_scatter(eval_df, W=48.0, k=8)
        all_hits += sg_hits + gs_hits

    # 4. CLUSTER & SCORE ALERTS
    log(f"Stage 4: Merging {len(all_hits):,} detector hits via shared edge Union-Find (max 300 accounts)...")
    t0 = time.time()
    clusters = merge_hits(all_hits, eval_df, max_accounts=300)
    log(f"Clusters created in {time.time() - t0:.2f}s | Total clusters: {len(clusters):,}")

    log("Stage 4b: Scoring and ranking alerts...")
    alerts = clusters_to_alerts(clusters, eval_df, id_prefix=eval_split_name[0].upper())
    alerts.sort(key=lambda a: a["risk_score"], reverse=True)
    log(f"Generated {len(alerts):,} ranked alerts.")

    # 5. SAVE ALERTS AND METRICS
    alerts_save_path = ROOT / f"alerts_{eval_split_name}.json"
    log(f"Stage 5: Saving alerts to {alerts_save_path}...")
    save_alerts(alerts, str(alerts_save_path))
    if eval_split_name == "val":
        save_alerts(alerts, str(ROOT / "alerts.json"))

    # 6. EVALUATION
    log("Stage 6: Evaluating PRF and Ring Recall...")
    # Transaction level: strictly eid in alert's edges
    mt_flagged_eids = flagged_eids_from_alerts(alerts)
    bl_flagged_series = evaluate_baseline(eval_df, best_params, return_series=True)
    bl_flagged_eids = set(eval_df.loc[bl_flagged_series, "eid"].tolist())

    mt_tx_prf = prf(mt_flagged_eids, eval_df)
    bl_tx_prf = prf(bl_flagged_eids, eval_df)

    # Matched-budget baseline comparison
    # MuleTrace flags N transactions; take top N from baseline ranked by amount_usd
    budget_n = len(mt_flagged_eids)
    bl_flagged_df = eval_df.loc[bl_flagged_series]
    bl_matched_top_n = bl_flagged_df.sort_values("amount_usd", ascending=False).head(budget_n)
    bl_matched_eids = set(bl_matched_top_n["eid"].tolist())
    bl_matched_prf = prf(bl_matched_eids, eval_df)

    # Account level: accounts involved in alerts
    mt_flagged_accounts = flagged_accounts_from_alerts(alerts)
    bl_flagged_accounts = set(eval_df.loc[eval_df["eid"].isin(bl_flagged_eids), "src"].tolist()) | \
                          set(eval_df.loc[eval_df["eid"].isin(bl_flagged_eids), "dst"].tolist())

    mt_acc_prf = account_level_prf(mt_flagged_accounts, eval_df)
    bl_acc_prf = account_level_prf(bl_flagged_accounts, eval_df)

    # Ring recall:
    # Patterns file
    patterns_file = str(ROOT / "data" / "HI-Small_Patterns.txt")
    raw_rings = parse_patterns(patterns_file)
    eval_ts_min = eval_df["ts"].min()
    eval_ts_max = eval_df["ts"].max()

    # Strict rings: both first and last timestamps fall inside split
    strict_rings = [
        r for r in raw_rings
        if r.get("first_ts") is not None and r.get("last_ts") is not None
        and r["first_ts"] >= eval_ts_min and r["last_ts"] <= eval_ts_max
    ]
    # Relaxed rings: first timestamp falls inside split
    relaxed_rings = [
        r for r in raw_rings
        if r.get("first_ts") is not None
        and r["first_ts"] >= eval_ts_min and r["first_ts"] <= eval_ts_max
    ]
    log(f"Rings strictly inside {eval_split_name.upper()} split window: {len(strict_rings)} (out of {len(raw_rings)} total)")
    log(f"Relaxed rings (first tx inside split): {len(relaxed_rings)}")

    rr_strict = ring_recall(alerts, strict_rings, threshold=0.70)
    rr_relaxed = ring_recall(alerts, relaxed_rings, threshold=0.70)

    # Per-ring best coverage fraction for relaxed set
    relaxed_ring_details = []
    for idx, r in enumerate(relaxed_rings, 1):
        r_accs = set(r.get("accounts", []))
        best_cov = 0.0
        best_alert_id = None
        if r_accs:
            for a in alerts:
                a_accs = set(a.get("accounts", []))
                cov = len(r_accs & a_accs) / len(r_accs)
                if cov > best_cov:
                    best_cov = cov
                    best_alert_id = a.get("alert_id")
        relaxed_ring_details.append({
            "ring_index": idx,
            "type": r.get("type", "UNKNOWN"),
            "accounts_count": len(r_accs),
            "best_coverage_fraction": round(best_cov, 4),
            "best_coverage_pct": f"{best_cov * 100:.1f}%",
            "found_70pct": best_cov >= 0.70,
            "best_alert_id": best_alert_id,
        })

    # Ranked alert metrics for top 50, 100, 250, 500 alerts
    ranked_metrics = []
    for k_cut in [50, 100, 250, 500]:
        sub_alerts = alerts[:k_cut]
        sub_eids: set[int] = set()
        for a in sub_alerts:
            for e in a.get("edges", []):
                if "eid" in e:
                    sub_eids.add(e["eid"])
        sub_df = eval_df[eval_df["eid"].isin(sub_eids)]
        illicit_cov = int(sub_df["label"].sum()) if not sub_df.empty else 0
        tx_prec = illicit_cov / len(sub_eids) if sub_eids else 0.0

        # Rings touched: at least 50% accounts in any single alert within top K
        touched_rings_strict = 0
        for r in strict_rings:
            r_accs = set(r.get("accounts", []))
            if not r_accs:
                continue
            for a in sub_alerts:
                if len(r_accs & set(a.get("accounts", []))) / len(r_accs) >= 0.50:
                    touched_rings_strict += 1
                    break

        touched_rings_relaxed = 0
        for r in relaxed_rings:
            r_accs = set(r.get("accounts", []))
            if not r_accs:
                continue
            for a in sub_alerts:
                if len(r_accs & set(a.get("accounts", []))) / len(r_accs) >= 0.50:
                    touched_rings_relaxed += 1
                    break

        ranked_metrics.append({
            "top_k_alerts": k_cut,
            "flagged_transactions": len(sub_eids),
            "illicit_covered": illicit_cov,
            "tx_precision": round(tx_prec, 6),
            "rings_touched_strict": f"{touched_rings_strict}/{len(strict_rings)}",
            "rings_touched_relaxed": f"{touched_rings_relaxed}/{len(relaxed_rings)}",
        })

    # 7. ALERT SIZE DISTRIBUTION & TOP 5 ALERTS
    alert_sizes = [len(a["accounts"]) for a in alerts] if alerts else [0]
    size_min = int(np.min(alert_sizes)) if alerts else 0
    size_median = float(np.median(alert_sizes)) if alerts else 0.0
    size_p90 = float(np.percentile(alert_sizes, 90)) if alerts else 0.0
    size_max = int(np.max(alert_sizes)) if alerts else 0

    top_5_alerts = sorted(alerts, key=lambda a: len(a["accounts"]), reverse=True)[:5]

    # Save results json
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_save_path = results_dir / f"run_{eval_split_name}.json"

    run_results = {
        "split": eval_split_name,
        "n_rows": len(eval_df),
        "illicit_rows": int(eval_df["label"].sum()),
        "n_alerts": len(alerts),
        "alert_size_distribution": {
            "min_accounts": size_min,
            "median_accounts": size_median,
            "p90_accounts": size_p90,
            "max_accounts": size_max,
        },
        "top_5_alerts": [
            {
                "alert_id": a["alert_id"],
                "risk_score": a["risk_score"],
                "typology": a["typology"],
                "n_accounts": len(a["accounts"]),
                "n_edges": len(a["edges"]),
                "total_amount_usd": a["evidence"]["total_amount_usd"],
            }
            for a in top_5_alerts
        ],
        "detector_stats": detector_stats,
        "baseline_tuned_params": best_params,
        "baseline_evaluation": {
            "train_tuned": bl_train,
            "train_plain_10k": plain_train,
            "val_tuned": bl_eval,
            "val_plain_10k": plain_eval,
        },
        "muletrace_tx_prf": mt_tx_prf,
        "baseline_tx_prf": bl_tx_prf,
        "matched_budget_baseline": {
            "budget_n": budget_n,
            "muletrace_prf": mt_tx_prf,
            "baseline_matched_prf": bl_matched_prf,
        },
        "ranked_alert_metrics": ranked_metrics,
        "ring_recall_strict": rr_strict,
        "ring_recall_relaxed": rr_relaxed,
        "relaxed_ring_details": relaxed_ring_details,
    }

    with open(results_save_path, "w") as f:
        json.dump(run_results, f, indent=2, default=str)
    log(f"Results saved to {results_save_path}")

    # -----------------------------------------------------------------------
    # PRINT SUMMARY TABLES AND DETAILED OUTPUT
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  MULETRACE EVALUATION SUMMARY — {eval_split_name.upper()} SPLIT")
    print("=" * 70)

    print(f"\n1. ALERTS SUMMARY (Edge-level merging, max 300 accounts/alert):")
    print(f"   Total alerts generated: {len(alerts):,}")
    print(f"   Alert size distribution (accounts per alert):")
    print(f"     Min: {size_min} | Median: {size_median:.1f} | P90: {size_p90:.1f} | Max: {size_max}")

    print(f"\n2. TOP 5 BIGGEST ALERTS (by account count):")
    for i, a in enumerate(top_5_alerts, 1):
        print(f"   #{i} [{a['alert_id']}] Typology: {a['typology']:<28} | Score: {a['risk_score']:>5.1f} | Accounts: {len(a['accounts']):>4} | Edges: {len(a['edges']):>4} | Total USD: ${a['evidence']['total_amount_usd']:>12,.2f}")

    print(f"\n3. PER-DETECTOR BREAKDOWN ({eval_split_name.upper()} split):")
    det_rows = []
    for dname, stats in detector_stats.items():
        det_rows.append({
            "detector": dname,
            "hits": stats["hits"],
            "tx_covered": stats["covered_transactions"],
            "tx_precision": round(stats["precision"], 4),
            "time_sec": round(stats["time_s"], 2),
        })
    print_table(det_rows)

    print(f"\n4. TRANSACTION-LEVEL PRF ({eval_split_name.upper()} split):")
    print("   * Note: MuleTrace tx is flagged ONLY if its eid is in an alert edge *")
    print_table([
        {"system": "Baseline (Tuned)",   **bl_tx_prf, "n_flagged": len(bl_flagged_eids)},
        {"system": "MuleTrace (Edges)", **mt_tx_prf, "n_flagged": len(mt_flagged_eids)},
    ])

    print(f"\n5. MATCHED-BUDGET BASELINE COMPARISON (Budget N = {budget_n:,} transactions):")
    print_table([
        {"system": f"Baseline (Top {budget_n:,} by USD)", **bl_matched_prf, "n_flagged": len(bl_matched_eids)},
        {"system": f"MuleTrace (Edge-flagged N)",         **mt_tx_prf,      "n_flagged": len(mt_flagged_eids)},
    ])

    print(f"\n6. RANKED-ALERT METRICS (Top K Alerts by Risk Score):")
    print_table(ranked_metrics)

    print(f"\n7. ACCOUNT-LEVEL PRF ({eval_split_name.upper()} split):")
    print_table([
        {"system": "Baseline (Tuned)",   **bl_acc_prf},
        {"system": "MuleTrace (Alerts)", **mt_acc_prf},
    ])

    print(f"\n8. STRICT RING RECALL (Single Alert >= 70% accounts, rings strictly inside split):")
    print(f"   Overall recall: {rr_strict['found']}/{rr_strict['total']} ({rr_strict['overall_recall'] * 100:.2f}%)")
    strict_rr_rows = []
    for rtype, stats in rr_strict.get("per_type", {}).items():
        strict_rr_rows.append({
            "ring_type": rtype,
            "found": stats["found"],
            "total": stats["total"],
            "recall": f"{stats['recall'] * 100:.1f}%",
        })
    print_table(strict_rr_rows)

    print(f"\n9. RELAXED RING RECALL & NEAR MISSES (Rings starting inside split, N={len(relaxed_rings)}):")
    print(f"   Overall recall (>=70%): {rr_relaxed['found']}/{rr_relaxed['total']} ({rr_relaxed['overall_recall'] * 100:.2f}%)")
    print("   Per-ring details in relaxed window:")
    print_table(relaxed_ring_details)

    total_time = time.time() - t_start_total
    log(f"\nPipeline run completed successfully in {total_time:.1f}s.")


if __name__ == "__main__":
    main()
