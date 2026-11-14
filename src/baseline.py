import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

def compute_daily_stats(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute daily transaction count and daily total amount_usd per sender account.
    """
    day = df['ts'].dt.floor('D')
    daily_n = df.groupby([day, 'src'])['amount_usd'].transform('count').to_numpy(dtype=float)
    daily_sum = df.groupby([day, 'src'])['amount_usd'].transform('sum').to_numpy(dtype=float)
    return daily_n, daily_sum

def baseline(
    df: pd.DataFrame,
    amt_thr: float,
    daily_n: float,
    daily_sum: float
) -> pd.Series:
    """
    Flag transactions where:
      - amount_usd > amt_thr, OR
      - sender has > daily_n transactions that day, OR
      - sender's daily total amount_usd > daily_sum.
    Returns a boolean pd.Series indexed matching df.
    """
    amt = df['amount_usd'].to_numpy(dtype=float)
    daily_counts, daily_totals = compute_daily_stats(df)
    
    flagged = (amt > amt_thr) | (daily_counts > daily_n) | (daily_totals > daily_sum)
    return pd.Series(flagged, index=df.index, name="baseline_flagged")

def tune_baseline(
    train: pd.DataFrame,
    amt_grid: Optional[List[float]] = None,
    n_grid: Optional[List[float]] = None,
    sum_grid: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Grid-search thresholds (amt_thr, daily_n, daily_sum) to maximise F1
    on the train split. Never uses or sees the validation or test split.
    """
    if amt_grid is None:
        amt_grid = [10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, float('inf')]
    if n_grid is None:
        n_grid = [2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, float('inf')]
    if sum_grid is None:
        sum_grid = [50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000, float('inf')]

    y_train = (train['label'] == 1).to_numpy(dtype=bool)
    amt_train = train['amount_usd'].to_numpy(dtype=float)
    train_n, train_sum = compute_daily_stats(train)

    best_f1 = -1.0
    best_params: Dict[str, Any] = {
        'amt_thr': float('inf'),
        'daily_n': float('inf'),
        'daily_sum': float('inf'),
        'f1': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'tp': 0,
        'fp': 0,
        'fn': int(np.sum(y_train))
    }

    for a in amt_grid:
        cond_a = amt_train > a
        for n in n_grid:
            cond_n = cond_a | (train_n > n)
            for s in sum_grid:
                pred = cond_n | (train_sum > s)
                
                tp = int(np.count_nonzero(pred & y_train))
                fp = int(np.count_nonzero(pred & ~y_train))
                fn = int(np.count_nonzero(~pred & y_train))
                
                if tp == 0:
                    f1 = 0.0
                    prec = 0.0
                    rec = 0.0
                else:
                    prec = tp / (tp + fp)
                    rec = tp / (tp + fn)
                    f1 = (2.0 * prec * rec) / (prec + rec)

                # Prioritize highest F1; break ties with higher precision to minimize false alarms
                if f1 > best_f1 or (abs(f1 - best_f1) < 1e-9 and prec > best_params['precision']):
                    best_f1 = f1
                    best_params = {
                        'amt_thr': a,
                        'daily_n': n,
                        'daily_sum': s,
                        'f1': f1,
                        'precision': prec,
                        'recall': rec,
                        'tp': tp,
                        'fp': fp,
                        'fn': fn
                    }

    return best_params

def evaluate_baseline(
    df: pd.DataFrame,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate fixed baseline parameters on a given split (e.g. validation).
    """
    y_true = (df['label'] == 1).to_numpy(dtype=bool)
    flagged = baseline(
        df,
        amt_thr=params['amt_thr'],
        daily_n=params['daily_n'],
        daily_sum=params['daily_sum']
    ).to_numpy(dtype=bool)

    tp = int(np.count_nonzero(flagged & y_true))
    fp = int(np.count_nonzero(flagged & ~y_true))
    fn = int(np.count_nonzero(~flagged & y_true))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'flagged_count': int(np.sum(flagged)),
        'total_count': len(df)
    }

if __name__ == "__main__":
    from src.load import load, time_split
    
    print("Evaluating Baseline on 200,000 rows slice...")
    df = load("data/HI-Small_Trans.csv", nrows=200000)
    train, val, test = time_split(df)
    
    print(f"Train split: {len(train):,} rows ({train['label'].sum()} illicit)")
    print(f"Val split:   {len(val):,} rows ({val['label'].sum()} illicit)")
    print(f"Test split:  {len(test):,} rows ({test['label'].sum()} illicit)")
    
    print("\nTuning baseline on train split...")
    best_params = tune_baseline(train)
    print("Best parameters found on Train:")
    for k, v in best_params.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}" if v < 1e6 else f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")
            
    print("\nEvaluating tuned baseline on Val split...")
    val_results = evaluate_baseline(val, best_params)
    print("Validation Results:")
    for k, v in val_results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
