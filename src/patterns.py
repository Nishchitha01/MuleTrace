import os
from typing import List, Dict, Any, Set
from collections import Counter
import pandas as pd

def parse_patterns(path: str = "data/HI-Small_Patterns.txt") -> List[Dict[str, Any]]:
    """
    Parse HI-Small_Patterns.txt.
    Each ring starts with 'BEGIN LAUNDERING ATTEMPT - <TYPE>' and ends with 'END LAUNDERING ATTEMPT'.
    Returns a list of dicts:
      - 'type': ring typology name (e.g. 'CYCLE', 'FAN-OUT')
      - 'accounts': set of account identifiers formatted as bank_account
      - 'raw_accounts': set of raw account numbers
      - 'first_ts': first timestamp (pd.Timestamp)
      - 'last_ts': last timestamp (pd.Timestamp)
      - 'transactions': list of raw transaction lines
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Patterns file not found: {path}")

    rings = []
    current_ring = None

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                # Header format: BEGIN LAUNDERING ATTEMPT - <TYPE>[: optional details]
                after_dash = line.split(" - ", 1)[1] if " - " in line else line.replace("BEGIN LAUNDERING ATTEMPT", "").strip()
                ring_type = after_dash.split(":", 1)[0].strip()
                
                current_ring = {
                    "type": ring_type,
                    "accounts": set(),
                    "raw_accounts": set(),
                    "timestamps": [],
                    "transactions": []
                }
            elif line.startswith("END LAUNDERING ATTEMPT"):
                if current_ring is not None:
                    if current_ring["timestamps"]:
                        current_ring["first_ts"] = min(current_ring["timestamps"])
                        current_ring["last_ts"] = max(current_ring["timestamps"])
                    else:
                        current_ring["first_ts"] = None
                        current_ring["last_ts"] = None
                    del current_ring["timestamps"]
                    rings.append(current_ring)
                    current_ring = None
            else:
                if current_ring is not None:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 5:
                        ts = pd.to_datetime(parts[0], format="%Y/%m/%d %H:%M")
                        from_bank = parts[1]
                        src_acc = parts[2]
                        to_bank = parts[3]
                        dst_acc = parts[4]

                        src = f"{from_bank}_{src_acc}"
                        dst = f"{to_bank}_{dst_acc}"

                        current_ring["accounts"].add(src)
                        current_ring["accounts"].add(dst)
                        current_ring["raw_accounts"].add(src_acc)
                        current_ring["raw_accounts"].add(dst_acc)
                        current_ring["timestamps"].append(ts)
                        current_ring["transactions"].append(line)

    return rings

def count_rings_by_type(rings: List[Dict[str, Any]]) -> Dict[str, int]:
    """Return dictionary of ring counts per typology type."""
    counter = Counter(r["type"] for r in rings)
    return dict(sorted(counter.items(), key=lambda x: x[1], reverse=True))

if __name__ == "__main__":
    patterns_file = "data/HI-Small_Patterns.txt"
    rings = parse_patterns(patterns_file)
    counts = count_rings_by_type(rings)
    
    print(f"Total Laundering Rings Parsed: {len(rings)}\n")
    print("Ring Counts Per Type:")
    print("-" * 35)
    for r_type, count in counts.items():
        print(f"  {r_type:<20} : {count:>4}")
    print("-" * 35)
    print(f"  {'TOTAL':<20} : {len(rings):>4}")
# ---- normalise ring account IDs so they match src/dst in load.py ----
def _norm_acc(a):
    bank, _, acc = a.partition("_")
    try:
        bank = str(int(bank))      # "020" -> "20", "00410" -> "410"
    except ValueError:
        pass
    return f"{bank}_{acc}"

_raw_parse_patterns = parse_patterns

def parse_patterns(path):
    rings = _raw_parse_patterns(path)
    for r in rings:
        r["accounts"] = {_norm_acc(a) for a in r["accounts"]}
    return rings