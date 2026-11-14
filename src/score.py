"""
score.py — Score, rank, and serialise clusters into alerts.

P10: rule_score, typology, and alert serialisation.

rule_score (0–100) = 100 × (
    0.30 × structure_score   # pattern diversity and cycle presence
  + 0.25 × amount_score      # log-normalised total amount
  + 0.20 × speed_score       # how fast money moves (low duration = suspicious)
  + 0.15 × diversity_score   # number of distinct patterns
  + 0.10 × size_score        # number of accounts involved
)
"""
from __future__ import annotations

import json
import math
import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.cluster import merge_hits


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

# Approximate normalization anchors (tuned on 200k slice)
_MAX_LOG_AMOUNT   = math.log1p(10_000_000)   # ~$10M ceiling
_MAX_ACCOUNTS     = 50
_MIN_DURATION_SUSPICIOUS = 0.5               # hours — sub-hour is very suspicious
_MAX_DURATION_FOR_SPEED  = 168.0             # hours — 1-week window cap


def _structure_score(patterns: set[str]) -> float:
    """Higher if cycle is present; bonus for chain; baseline for fan."""
    if "cycle" in patterns:
        return 1.0
    if "chain" in patterns and ("fan_in" in patterns or "fan_out" in patterns):
        return 0.75
    if "chain" in patterns:
        return 0.65
    if "fan_in" in patterns and "fan_out" in patterns:
        return 0.70
    if "scatter_gather" in patterns or "gather_scatter" in patterns:
        return 0.80
    return 0.40          # single fan pattern


def _amount_score(total_usd: float) -> float:
    return min(math.log1p(total_usd) / _MAX_LOG_AMOUNT, 1.0)


def _speed_score(duration_hours: float) -> float:
    """Very fast pass-through (< 0.5 h) scores 1.0; slow (> 168 h) scores 0."""
    if duration_hours <= _MIN_DURATION_SUSPICIOUS:
        return 1.0
    if duration_hours >= _MAX_DURATION_FOR_SPEED:
        return 0.0
    return 1.0 - (duration_hours - _MIN_DURATION_SUSPICIOUS) / (
        _MAX_DURATION_FOR_SPEED - _MIN_DURATION_SUSPICIOUS
    )


def _diversity_score(patterns: set[str]) -> float:
    n = len(patterns)
    return min(n / 4.0, 1.0)      # 4+ patterns → full score


def _size_score(n_accounts: int) -> float:
    return min(n_accounts / _MAX_ACCOUNTS, 1.0)


def rule_score(cluster: dict) -> float:
    s  = _structure_score(cluster["patterns"])
    a  = _amount_score(cluster["total_amount_usd"])
    sp = _speed_score(cluster["duration_hours"])
    d  = _diversity_score(cluster["patterns"])
    sz = _size_score(len(cluster["accounts"]))
    raw = 0.30 * s + 0.25 * a + 0.20 * sp + 0.15 * d + 0.10 * sz
    return round(raw * 100, 2)


# ---------------------------------------------------------------------------
# Alert serialisation
# ---------------------------------------------------------------------------

def _t_to_iso(t_hours: float, epoch: datetime.datetime) -> str:
    """Convert t (hours since first tx) back to ISO-8601."""
    dt = epoch + datetime.timedelta(hours=t_hours)
    return dt.strftime("%Y-%m-%dT%H:%M")


def clusters_to_alerts(
    clusters: list[dict],
    df: pd.DataFrame,
    id_prefix: str = "A",
) -> list[dict]:
    """
    Convert scored clusters into alert dicts matching docs/alert_format.json.
    df is used to recover the epoch timestamp.
    """
    if df.empty or not clusters:
        return []

    epoch = df["ts"].min()   # datetime corresponding to t=0

    alerts: list[dict] = []
    for rank, cl in enumerate(clusters, start=1):
        rs = rule_score(cl)
        # risk_score = rule_score for now (P17 adds learned model probability)
        alert_id = f"{id_prefix}-{rank:04d}"

        # Build edge list with ISO timestamps
        edges_out = []
        for e in cl["edges"]:
            edges_out.append(
                {
                    "eid": e["eid"],           # kept for downstream evaluation
                    "from": e["from"],
                    "to": e["to"],
                    "time": _t_to_iso(e["time"], epoch),
                    "amount_usd": round(e["amount_usd"], 2),
                }
            )

        # Evidence block
        n_edges = len(cl["edges"])
        evidence: dict = {
            "total_amount_usd": round(cl["total_amount_usd"], 2),
            "duration_hours": round(cl["duration_hours"], 2),
            "n_edges": n_edges,
            "n_accounts": len(cl["accounts"]),
        }
        if "cycle" in cl["patterns"]:
            evidence["cycle_length"] = min(
                n_edges, 5
            )  # rough proxy; exact length in P8

        alert: dict = {
            "alert_id": alert_id,
                        "risk_score": round(25 * max({"cycle": 3, "chain": 2, "fan_in": 1, "fan_out": 0}.get(p, 0) for p in cl["patterns"]) + 0.25 * rs, 2),          # = rule_score now; P17 adds learned score
            "rule_score": rs,
            "typology": cl["typology"],
            "patterns": sorted(cl["patterns"]),
            "accounts": sorted(cl["accounts"]),
            "edges": edges_out,
            "evidence": evidence,
            "risk_breakdown": [],      # P18 adds SHAP values
            "key_players": [],         # P19 adds mule roles
            "summary": None,           # P12 adds LLM summary
        }
        alerts.append(alert)

    return alerts


def save_alerts(alerts: list[dict], path: str) -> None:
    """Write alerts list to JSON (sets serialised as sorted lists)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(alerts, f, indent=2, default=_json_default)


def _json_default(obj):
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, float) and math.isnan(obj):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# Top-level: run detectors on a split and return alerts
# ---------------------------------------------------------------------------

def run_detectors_and_score(
    split_df: pd.DataFrame,
    full_df: pd.DataFrame,
    id_prefix: str = "A",
    fan_k: int = 8,
    fan_W: float = 48.0,
    chain_D: float = 6.0,
    chain_keep: float = 0.70,
    cycle_max_len: int = 5,
    cycle_W: float = 168.0,
    enable_composites: bool = False,
) -> list[dict]:
    """Run all detectors on split_df, cluster, score, and return alert list."""
    from src.detectors import fan, chains, temporal_cycles

    hits: list[dict] = []
    hits += fan(split_df, key="src", other="dst", k=fan_k, W=fan_W)
    hits += fan(split_df, key="dst", other="src", k=fan_k, W=fan_W)
    hits += chains(split_df, D=chain_D, keep=chain_keep)
    hits += temporal_cycles(split_df, max_len=cycle_max_len, W=cycle_W)
    if enable_composites:
        from src.detectors import scatter_gather, gather_scatter
        hits += scatter_gather(split_df, W=fan_W, k=fan_k)
        hits += gather_scatter(split_df, W=fan_W, k=fan_k)

    clusters = merge_hits(hits, full_df)
    alerts = clusters_to_alerts(clusters, full_df, id_prefix=id_prefix)
    # Sort by risk_score descending
    alerts.sort(key=lambda a: a["risk_score"], reverse=True)
    return alerts
