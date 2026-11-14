import pandas as pd
import numpy as np
from typing import Set, List, Dict, Any, Union, Optional
from collections import defaultdict

def prf(
    flagged_eids: Union[Set[int], List[int], np.ndarray, pd.Series],
    df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Transaction-level evaluation: precision, recall, and F1.
    
    Parameters:
      flagged_eids: Collection of flagged transaction row IDs (eid) or boolean mask
      df: DataFrame containing 'eid' and ground truth 'label' (1 for illicit, 0 for legit)
      
    Returns:
      Dict with precision, recall, f1, tp, fp, fn, tn, flagged_count, total_count.
    """
    if isinstance(flagged_eids, (pd.Series, np.ndarray)) and flagged_eids.dtype == bool:
        flagged_mask = flagged_eids
        flagged_set = set(df.loc[flagged_mask, 'eid'])
    else:
        flagged_set = set(flagged_eids)

    y_true = (df['label'] == 1).to_numpy(dtype=bool)
    eids = df['eid'].to_numpy()
    flagged_mask = np.isin(eids, list(flagged_set))

    tp = int(np.count_nonzero(flagged_mask & y_true))
    fp = int(np.count_nonzero(flagged_mask & ~y_true))
    fn = int(np.count_nonzero(~flagged_mask & y_true))
    tn = int(np.count_nonzero(~flagged_mask & ~y_true))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'flagged_count': int(np.count_nonzero(flagged_mask)),
        'total_count': len(df)
    }

def account_level_prf(
    flagged_accounts: Union[Set[str], List[str], Set[int], List[int]],
    df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Account-level evaluation: precision, recall, and F1.
    
    Illicit accounts are those participating as src or dst in any transaction with label == 1.
    If integer eids are passed, accounts are extracted from the flagged transactions in df.
    """
    # Identify ground truth illicit accounts
    illicit_txs = df[df['label'] == 1]
    illicit_accounts = set(illicit_txs['src']).union(set(illicit_txs['dst']))
    
    all_accounts = set(df['src']).union(set(df['dst']))

    # Handle input if passed as eids or account strings
    flagged_list = list(flagged_accounts)
    if flagged_list and isinstance(flagged_list[0], (int, np.integer)):
        flagged_df = df[df['eid'].isin(set(flagged_list))]
        flagged_acc_set = set(flagged_df['src']).union(set(flagged_df['dst']))
    else:
        flagged_acc_set = set(flagged_accounts)

    tp_accounts = flagged_acc_set & illicit_accounts
    fp_accounts = flagged_acc_set - illicit_accounts
    fn_accounts = illicit_accounts - flagged_acc_set
    tn_accounts = (all_accounts - illicit_accounts) - flagged_acc_set

    tp = len(tp_accounts)
    fp = len(fp_accounts)
    fn = len(fn_accounts)
    tn = len(tn_accounts)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'flagged_accounts': len(flagged_acc_set),
        'total_accounts': len(all_accounts)
    }

def ring_recall(
    alerts: List[Union[Dict[str, Any], Set[str], List[str]]],
    rings: List[Dict[str, Any]],
    threshold: float = 0.70
) -> Dict[str, Any]:
    """
    Calculate ring recall.
    A ring counts as found if at least 70% of its accounts appear in any alert.
    
    Returns overall recall and recall per ring type.
    """
    # Pre-extract account sets for alerts
    alert_account_sets = []
    for a in alerts:
        if isinstance(a, dict):
            alert_account_sets.append(set(a.get('accounts', [])))
        else:
            alert_account_sets.append(set(a))

    type_stats = defaultdict(lambda: {'found': 0, 'total': 0})
    total_found = 0
    total_rings = len(rings)

    for ring in rings:
        r_type = ring.get('type', 'UNKNOWN')
        ring_accounts = set(ring.get('accounts', []))
        
        type_stats[r_type]['total'] += 1
        if not ring_accounts:
            continue

        # Check if at least 70% of this ring's accounts appear in any single alert
        is_found = False
        ring_size = len(ring_accounts)
        
        for alert_accs in alert_account_sets:
            overlap = len(ring_accounts & alert_accs)
            if (overlap / ring_size) >= threshold:
                is_found = True
                break

        if is_found:
            total_found += 1
            type_stats[r_type]['found'] += 1

    overall_recall = total_found / total_rings if total_rings > 0 else 0.0

    per_type_result = {}
    for r_type, stats in sorted(type_stats.items()):
        rec = stats['found'] / stats['total'] if stats['total'] > 0 else 0.0
        per_type_result[r_type] = {
            'found': stats['found'],
            'total': stats['total'],
            'recall': float(rec)
        }

    return {
        'overall_recall': float(overall_recall),
        'found': total_found,
        'total': total_rings,
        'per_type': per_type_result
    }

def print_table(results: Union[List[Dict[str, Any]], pd.DataFrame, Dict[str, Dict[str, Any]]]) -> None:
    """
    Print a cleanly formatted ASCII comparison table.
    """
    if isinstance(results, pd.DataFrame):
        df_table = results.copy()
    elif isinstance(results, list):
        df_table = pd.DataFrame(results)
    elif isinstance(results, dict):
        # Convert dict of dicts to dataframe
        records = []
        for k, v in results.items():
            row = {'Method': k}
            row.update(v)
            records.append(row)
        df_table = pd.DataFrame(records)
    else:
        raise ValueError("Unsupported results format for print_table.")

    # Format numeric columns to 4 decimal places
    formatted_df = df_table.copy()
    for col in formatted_df.columns:
        if pd.api.types.is_float_dtype(formatted_df[col]):
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}")

    # Build ASCII table
    headers = [str(c) for c in formatted_df.columns]
    rows = [[str(item) for item in row] for row in formatted_df.values]
    
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(val))

    separator = "+" + "+".join(["-" * (w + 2) for w in col_widths]) + "+"
    header_str = "| " + " | ".join([h.ljust(col_widths[i]) for i, h in enumerate(headers)]) + " |"

    print(separator)
    print(header_str)
    print(separator)
    for row in rows:
        row_str = "| " + " | ".join([val.ljust(col_widths[i]) for i, val in enumerate(row)]) + " |"
        print(row_str)
    print(separator)

if __name__ == "__main__":
    print("Testing evaluate.py on tiny hand-made example:")
    df_toy = pd.DataFrame({
        'eid': [0, 1, 2, 3, 4],
        'src': ['ACC_A', 'ACC_A', 'ACC_B', 'ACC_C', 'ACC_D'],
        'dst': ['ACC_B', 'ACC_C', 'ACC_D', 'ACC_A', 'ACC_B'],
        'label': [0, 1, 0, 1, 0]
    })
    flagged = [1, 2]
    tx_res = prf(flagged, df_toy)
    print("\nTransaction-level PRF:")
    print(tx_res)

    acc_res = account_level_prf(['ACC_A', 'ACC_B'], df_toy)
    print("\nAccount-level PRF:")
    print(acc_res)

    toy_alerts = [
        {'alert_id': 'A1', 'accounts': ['ACC_A', 'ACC_B', 'ACC_C']},
        {'alert_id': 'A2', 'accounts': ['ACC_D', 'ACC_E']}
    ]
    toy_rings = [
        {'type': 'CYCLE', 'accounts': {'ACC_A', 'ACC_B', 'ACC_C'}},
        {'type': 'FAN-OUT', 'accounts': {'ACC_D', 'ACC_E', 'ACC_F', 'ACC_G'}}
    ]
    ring_res = ring_recall(toy_alerts, toy_rings)
    print("\nRing Recall:")
    print(ring_res)

    print("\nComparison Table:")
    print_table([
        {'Model': 'Baseline', 'Tx Precision': tx_res['precision'], 'Tx Recall': tx_res['recall'], 'Tx F1': tx_res['f1']},
        {'Model': 'Hand-Made', 'Tx Precision': 0.5000, 'Tx Recall': 0.5000, 'Tx F1': 0.5000}
    ])
