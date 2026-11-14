"""
cluster.py — Merge detector hits into connected-component clusters.

Merge criteria:
  - Hits are merged ONLY if they share at least one edge (the same eid),
    using Union-Find over hits via an eid-to-hits index.
  - Hits that only share accounts are NOT merged.
  - If a merged cluster exceeds 300 accounts, it is split back into its
    original hits, each becoming its own cluster.
  - Accounts, edges, total amount, duration, patterns, and typology are
    recomputed per cluster.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Union-Find (path-compressed)
# ---------------------------------------------------------------------------

class _UF:
    def __init__(self):
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path compression
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


# ---------------------------------------------------------------------------
# Typology mapping
# ---------------------------------------------------------------------------

_TYPOLOGY_PRIORITY = [
    # (frozenset of required patterns,  typology label)
    ({"cycle"},                              "Round-tripping"),
    ({"fan_in", "fan_out"},                  "Gather-scatter layering hub"),
    ({"scatter_gather"},                     "Scatter-gather layering"),
    ({"gather_scatter"},                     "Gather-scatter layering"),
    ({"chain"},                              "Layering through mule accounts"),
    ({"fan_out"},                            "Dispersion/structuring"),
    ({"fan_in"},                             "Aggregation of funds"),
]

def _map_typology(patterns: set[str]) -> str:
    for required, label in _TYPOLOGY_PRIORITY:
        if required <= patterns:          # required is a subset of patterns
            return label
    return "Suspicious activity"


# ---------------------------------------------------------------------------
# Cluster builder helper
# ---------------------------------------------------------------------------

def _build_cluster(hit_indices: list[int], hits: list[dict], eid_index: dict) -> Optional[dict]:
    all_eids: set[int] = set()
    all_patterns: set[str] = set()

    for idx in hit_indices:
        h = hits[idx]
        all_eids |= h["eids"]
        all_patterns.add(h["pattern"])

    all_accounts: set[str] = set()
    edge_rows: list[dict] = []
    for eid in all_eids:
        row = eid_index.get(eid)
        if row:
            all_accounts.add(row["src"])
            all_accounts.add(row["dst"])
            edge_rows.append(
                {
                    "eid": eid,
                    "from": row["src"],
                    "to": row["dst"],
                    "time": row["t"],
                    "amount_usd": row["amount_usd"],
                }
            )

    if not edge_rows:
        return None

    t_vals = [e["time"] for e in edge_rows]
    amt_vals = [e["amount_usd"] for e in edge_rows]
    total_amount = sum(amt_vals)
    duration_hours = max(t_vals) - min(t_vals)

    eid_str = ",".join(str(e) for e in sorted(all_eids))
    cid = "C-" + hashlib.md5(eid_str.encode()).hexdigest()[:8].upper()
    typology = _map_typology(all_patterns)

    return {
        "cluster_id": cid,
        "accounts": all_accounts,
        "eids": all_eids,
        "edges": edge_rows,
        "total_amount_usd": total_amount,
        "duration_hours": duration_hours,
        "patterns": all_patterns,
        "typology": typology,
    }


# ---------------------------------------------------------------------------
# Main clustering function
# ---------------------------------------------------------------------------

def merge_hits(hits: list[dict], df: pd.DataFrame, max_accounts: int = 300) -> list[dict]:
    """
    Merge detector hits into clusters.

    Hits are merged ONLY if they share at least one edge (same eid).
    Union-Find is performed over hit indices via an eid-to-hits index.
    After merging, any cluster with > max_accounts (default 300) is split
    back into its original hits, each becoming its own cluster.
    """
    if not hits:
        return []

    # Build eid -> row lookup for relevant eids only
    relevant_eids = set().union(*(h["eids"] for h in hits)) if hits else set()
    sub_df = df[df["eid"].isin(relevant_eids)]
    eid_index = sub_df.set_index("eid")[["src", "dst", "t", "amount_usd"]].to_dict("index")

    # Step 1: Build eid -> hits index
    eid_to_hits: dict[int, list[int]] = defaultdict(list)
    for i, h in enumerate(hits):
        for eid in h["eids"]:
            eid_to_hits[eid].append(i)

    # Step 2: Union-Find over hit indices (0..len(hits)-1)
    uf = _UF()
    for eid, hit_indices in eid_to_hits.items():
        if len(hit_indices) > 1:
            first = str(hit_indices[0])
            for other in hit_indices[1:]:
                uf.union(first, str(other))

    # Step 3: Group hits by component root
    comp_hits: dict[str, list[int]] = defaultdict(list)
    for i in range(len(hits)):
        root = uf.find(str(i))
        comp_hits[root].append(i)

    # Step 4: Build clusters, splitting any with > max_accounts back into individual hits
    clusters: list[dict] = []
    seen_cids: set[str] = set()

    for root, hit_indices in comp_hits.items():
        merged_cl = _build_cluster(hit_indices, hits, eid_index)
        if merged_cl is None:
            continue

        if len(merged_cl["accounts"]) > max_accounts:
            # Split back into individual hits
            for idx in hit_indices:
                single_cl = _build_cluster([idx], hits, eid_index)
                if single_cl is not None and single_cl["cluster_id"] not in seen_cids:
                    seen_cids.add(single_cl["cluster_id"])
                    clusters.append(single_cl)
        else:
            if merged_cl["cluster_id"] not in seen_cids:
                seen_cids.add(merged_cl["cluster_id"])
                clusters.append(merged_cl)

    # Sort by total_amount_usd descending as initial ranking
    clusters.sort(key=lambda c: c["total_amount_usd"], reverse=True)
    return clusters
