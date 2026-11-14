"""
detectors.py — Pattern detectors for MuleTrace.

P6: fan()           — fan-out / fan-in sliding-window detector
P7: chains()        — rapid pass-through mule chain detector
P8: temporal_cycles() — directed temporal cycle detector (len 3-5)
P9: scatter_gather() / gather_scatter() — composite detectors
"""

from __future__ import annotations

import time
from collections import defaultdict
from itertools import combinations
from typing import Literal

import pandas as pd


# ---------------------------------------------------------------------------
# P6 — Fan-out / Fan-in
# ---------------------------------------------------------------------------

def fan(
    df: pd.DataFrame,
    key: Literal["src", "dst"] = "src",
    other: Literal["src", "dst"] = "dst",
    k: int = 8,
    W: float = 48.0,
) -> list[dict]:
    """
    Detect fan-out (key='src') or fan-in (key='dst') hubs.

    For each hub account in the `key` column, slide a W-hour window over
    transactions sorted by t.  If the hub contacts >= k distinct counterparties
    in the `other` column within the window, flag it.

    Uses a sliding-window counter (O(n) per hub) for speed.

    Returns a list of dicts:
        {pattern, eids (set of int), hub (str)}
    """
    pattern_name = "fan_out" if key == "src" else "fan_in"
    hits: list[dict] = []

    sub = df[["eid", key, other, "t"]].sort_values([key, "t"])

    for hub, grp in sub.groupby(key, sort=False):
        t_vals = grp["t"].to_numpy()
        other_vals = grp[other].to_numpy()
        eid_vals = grp["eid"].to_numpy()
        n = len(t_vals)
        if n < k:
            continue

        # Sliding-window counter: O(n) per hub
        counter: dict[str, int] = {}
        distinct = 0
        left = 0

        for right in range(n):
            # Add right element
            ov = other_vals[right]
            if counter.get(ov, 0) == 0:
                distinct += 1
            counter[ov] = counter.get(ov, 0) + 1

            # Shrink window from left to stay within W hours
            while t_vals[right] - t_vals[left] > W:
                lv = other_vals[left]
                counter[lv] -= 1
                if counter[lv] == 0:
                    distinct -= 1
                left += 1

            # Check trigger
            if distinct >= k:
                hits.append(
                    {
                        "pattern": pattern_name,
                        "eids": set(eid_vals[left : right + 1].tolist()),
                        "hub": hub,
                    }
                )
                # Jump left to right+1 to avoid near-duplicate windows
                for i in range(left, right + 1):
                    lv = other_vals[i]
                    counter[lv] -= 1
                    if counter[lv] == 0:
                        distinct -= 1
                left = right + 1

    return hits


# ---------------------------------------------------------------------------
# P7 — Rapid pass-through chains
# ---------------------------------------------------------------------------

def chains(
    df: pd.DataFrame,
    D: float = 6.0,
    keep: float = 0.70,
    max_chain_len: int = 20,
    max_hits: int = 50000,
) -> list[dict]:
    """
    Detect mule chains: linked in→out transactions at the same account where:
      - out_t  ∈ (in_t, in_t + D]            (strict forward in time)
      - out_amount ∈ [keep * in_amount, 1.05 * in_amount]
      - out_dst ≠ in_src                      (no direct reversal)

    Connected linked transactions form chains; keep only chains of length ≥ 3.
    Uses iterative DFS to avoid stack overflow. Caps at max_hits results.

    Returns list of dicts: {pattern: 'chain', eids: set, hub: first_intermediate}
    """
    # Build quick lookup: account → sorted list of (t, amount_usd, eid, src, dst)
    incoming: dict[str, list] = defaultdict(list)
    outgoing: dict[str, list] = defaultdict(list)

    for row in df[["eid", "src", "dst", "t", "amount_usd"]].itertuples(index=False):
        incoming[row.dst].append((row.t, row.amount_usd, row.eid, row.src, row.dst))
        outgoing[row.src].append((row.t, row.amount_usd, row.eid, row.src, row.dst))

    # Build eid → dst map for hub lookup
    eid_to_dst: dict[int, str] = dict(zip(df["eid"], df["dst"]))

    # For each account, link each incoming tx to outgoing tx satisfying constraints
    child: dict[int, list[int]] = defaultdict(list)

    for acct in set(incoming) & set(outgoing):
        inc_list = sorted(incoming[acct], key=lambda x: x[0])
        out_list = sorted(outgoing[acct], key=lambda x: x[0])

        out_ptr = 0
        for in_t, in_amt, in_eid, in_src, _ in inc_list:
            while out_ptr < len(out_list) and out_list[out_ptr][0] <= in_t:
                out_ptr += 1
            for j in range(out_ptr, len(out_list)):
                out_t, out_amt, out_eid, _, out_dst = out_list[j]
                if out_t > in_t + D:
                    break
                if out_amt < keep * in_amt or out_amt > 1.05 * in_amt:
                    continue
                if out_dst == in_src:
                    continue
                child[in_eid].append(out_eid)

    # Root eids: eids that have children but are not a child of anything
    all_children: set[int] = {c for children in child.values() for c in children}
    root_eids: set[int] = set(child.keys()) - all_children

    hits: list[dict] = []
    seen_eid_sets: set[frozenset] = set()

    # Iterative DFS using an explicit stack
    # Stack items: (current_eid, path_so_far_as_tuple)
    for root in root_eids:
        if len(hits) >= max_hits:
            break
        stack = [(root, (root,))]
        while stack:
            cur_eid, path = stack.pop()
            if len(path) >= max_chain_len or cur_eid not in child:
                # Leaf: emit if long enough
                if len(path) >= 3:
                    fs = frozenset(path)
                    if fs not in seen_eid_sets:
                        seen_eid_sets.add(fs)
                        hub = eid_to_dst.get(path[0], "unknown")
                        hits.append({"pattern": "chain", "eids": set(path), "hub": hub})
                        if len(hits) >= max_hits:
                            break
                continue
            # Has children: push them (emit partial chain if no children match)
            pushed = False
            for next_eid in child[cur_eid]:
                stack.append((next_eid, path + (next_eid,)))
                pushed = True
            if not pushed and len(path) >= 3:
                fs = frozenset(path)
                if fs not in seen_eid_sets:
                    seen_eid_sets.add(fs)
                    hub = eid_to_dst.get(path[0], "unknown")
                    hits.append({"pattern": "chain", "eids": set(path), "hub": hub})

    return hits


# ---------------------------------------------------------------------------
# P8 — Temporal directed cycles
# ---------------------------------------------------------------------------

def temporal_cycles(
    df: pd.DataFrame,
    max_len: int = 5,
    W: float = 168.0,
    max_branch: int = 20,
) -> list[dict]:
    """
    Find directed cycles of length 3 to max_len where:
      - Timestamps strictly increase along the path
      - All hops within W hours of the first hop
      - Each hop's amount_usd is in [70%, 105%] of the first hop's amount

    Optimisation:
      1. Prune edges whose src never appears as dst (or vice versa).
      2. Cap branching factor to max_branch per node.

    Returns list of dicts: {pattern: 'cycle', eids: set, hub: start_account}
    """
    # --- Step 1: prune edges ---
    all_srcs = set(df["src"].unique())
    all_dsts = set(df["dst"].unique())
    # Keep only edges where src is also a dst AND dst is also a src
    eligible_srcs = all_srcs & all_dsts
    eligible_dsts = all_srcs & all_dsts

    sub = df[df["src"].isin(eligible_srcs) & df["dst"].isin(eligible_dsts)].copy()
    sub = sub.sort_values("t").reset_index(drop=True)

    if sub.empty:
        return []

    # Build adjacency: src → sorted list of (t, amount_usd, eid, dst)
    adj: dict[str, list] = defaultdict(list)
    for row in sub[["src", "dst", "t", "amount_usd", "eid"]].itertuples(index=False):
        adj[row.src].append((row.t, row.amount_usd, row.eid, row.dst))

    # Sort each adjacency list by t, cap branching factor
    for src in adj:
        adj[src].sort(key=lambda x: x[0])
        if len(adj[src]) > max_branch:
            adj[src] = adj[src][:max_branch]

    hits: list[dict] = []
    seen_cycles: set[frozenset] = set()

    # DFS: path = list of account names; path_eids = list of edge ids
    def dfs(
        node: str,
        start: str,
        path: list[str],
        path_eids: list[int],
        last_t: float,
        t0: float,
        amt0: float,
    ) -> None:
        if len(path) > max_len:
            return
        if node not in adj:
            return

        for t_next, amt_next, eid_next, dst_next in adj[node]:
            # Time must be strictly greater than last hop
            if t_next <= last_t:
                continue
            # All within W hours of start
            if t_next - t0 > W:
                break  # adj is sorted by t, so no point continuing
            # Amount ratio
            if amt_next < 0.70 * amt0 or amt_next > 1.05 * amt0:
                continue

            # Cycle found?
            if dst_next == start and len(path) >= 2:  # path already has start+intermediates
                cycle_eids = frozenset(path_eids + [eid_next])
                if cycle_eids not in seen_cycles:
                    seen_cycles.add(cycle_eids)
                    hits.append(
                        {
                            "pattern": "cycle",
                            "eids": set(cycle_eids),
                            "hub": start,
                        }
                    )
                continue

            # Avoid revisiting nodes in the same path (no sub-cycles)
            if dst_next in path:
                continue

            dfs(
                dst_next,
                start,
                path + [dst_next],
                path_eids + [eid_next],
                t_next,
                t0,
                amt0,
            )

    # Iterate over starting edges
    for src in list(adj.keys()):
        for t0, amt0, eid0, dst0 in adj[src]:
            if dst0 == src:
                continue  # self-loop
            dfs(dst0, src, [src, dst0], [eid0], t0, t0, amt0)

    return hits


# ---------------------------------------------------------------------------
# P9 — Composite detectors: scatter_gather / gather_scatter
# ---------------------------------------------------------------------------

def _build_fan_index(hits: list[dict], role: str) -> dict[str, list[dict]]:
    """Index fan hits by hub account for quick lookup."""
    index: dict[str, list[dict]] = defaultdict(list)
    for h in hits:
        if h["pattern"] == role:
            index[h["hub"]].append(h)
    return index


def scatter_gather(
    df: pd.DataFrame,
    W: float = 48.0,
    k: int = 8,
) -> list[dict]:
    """
    Detect scatter-gather: a fan-out from hub A and a fan-in to hub B share
    ≥ 2 intermediate accounts within the same W-hour window.

    pattern name: 'scatter_gather'
    hub: account A (the scatter hub)
    """
    fo_hits = fan(df, key="src", other="dst", k=k, W=W)
    fi_hits = fan(df, key="dst", other="src", k=k, W=W)

    if not fo_hits or not fi_hits:
        return []

    # For each fan-out hit, find the set of dst accounts (intermediaries)
    # For each fan-in hit, find the set of src accounts (intermediaries)
    # Match pairs that share ≥ 2 intermediaries

    fo_intermediaries: list[tuple[dict, set[str]]] = []
    for h in fo_hits:
        eids = h["eids"]
        intermediaries = set(df.loc[df["eid"].isin(eids), "dst"].tolist())
        fo_intermediaries.append((h, intermediaries))

    fi_intermediaries: list[tuple[dict, set[str]]] = []
    for h in fi_hits:
        eids = h["eids"]
        intermediaries = set(df.loc[df["eid"].isin(eids), "src"].tolist())
        fi_intermediaries.append((h, intermediaries))

    hits: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for fo_h, fo_inter in fo_intermediaries:
        for fi_h, fi_inter in fi_intermediaries:
            shared = fo_inter & fi_inter
            if len(shared) >= 2:
                pair = (fo_h["hub"], fi_h["hub"])
                if pair in seen:
                    continue
                seen.add(pair)
                hits.append(
                    {
                        "pattern": "scatter_gather",
                        "eids": fo_h["eids"] | fi_h["eids"],
                        "hub": fo_h["hub"],
                        "gather_hub": fi_h["hub"],
                        "shared_intermediaries": shared,
                    }
                )
    return hits


def gather_scatter(
    df: pd.DataFrame,
    W: float = 48.0,
    k: int = 8,
) -> list[dict]:
    """
    Detect gather-scatter: a fan-in to hub A and a fan-out from hub B share
    ≥ 2 intermediate accounts. Reverse of scatter_gather.

    pattern name: 'gather_scatter'
    hub: account A (the gather hub)
    """
    fi_hits = fan(df, key="dst", other="src", k=k, W=W)
    fo_hits = fan(df, key="src", other="dst", k=k, W=W)

    if not fi_hits or not fo_hits:
        return []

    fi_intermediaries: list[tuple[dict, set[str]]] = []
    for h in fi_hits:
        eids = h["eids"]
        intermediaries = set(df.loc[df["eid"].isin(eids), "src"].tolist())
        fi_intermediaries.append((h, intermediaries))

    fo_intermediaries: list[tuple[dict, set[str]]] = []
    for h in fo_hits:
        eids = h["eids"]
        intermediaries = set(df.loc[df["eid"].isin(eids), "dst"].tolist())
        fo_intermediaries.append((h, intermediaries))

    hits: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for fi_h, fi_inter in fi_intermediaries:
        for fo_h, fo_inter in fo_intermediaries:
            shared = fi_inter & fo_inter
            if len(shared) >= 2:
                pair = (fi_h["hub"], fo_h["hub"])
                if pair in seen:
                    continue
                seen.add(pair)
                hits.append(
                    {
                        "pattern": "gather_scatter",
                        "eids": fi_h["eids"] | fo_h["eids"],
                        "hub": fi_h["hub"],
                        "scatter_hub": fo_h["hub"],
                        "shared_intermediaries": shared,
                    }
                )
    return hits
