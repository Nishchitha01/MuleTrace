import pandas as pd
from typing import Optional, Tuple, Dict

# Static exchange rate dictionary to USD based on the IBM AML dataset values
STATIC_USD_RATES: Dict[str, float] = {
    'US Dollar': 1.0,
    'Bitcoin': 11874.4025,
    'Euro': 1.1718,
    'Australian Dollar': 0.7078,
    'Yuan': 0.1493,
    'Rupee': 0.0136,
    'Mexican Peso': 0.0473,
    'Yen': 0.0095,
    'UK Pound': 1.2917,
    'Ruble': 0.0129,
    'Canadian Dollar': 0.7580,
    'Swiss Franc': 1.0929,
    'Brazil Real': 0.1771,
    'Saudi Riyal': 0.2666,
    'Shekel': 0.2961,
}

COLS = [
    'ts', 'from_bank', 'src_acc', 'to_bank', 'dst_acc',
    'amt_recv', 'cur_recv', 'amount', 'cur_paid', 'fmt', 'label'
]

def load(path: str = "data/HI-Small_Trans.csv", nrows: Optional[int] = None, rates: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    Read HI-Small_Trans.csv, rename columns, parse timestamps,
    build bank-account identifiers, compute relative hours t,
    sort by t, assign eid row IDs, and normalise amount to amount_usd.
    """
    df = pd.read_csv(path, nrows=nrows, header=0, names=COLS)
    
    # Parse ts with format %Y/%m/%d %H:%M
    df['ts'] = pd.to_datetime(df['ts'], format='%Y/%m/%d %H:%M')
    
    # Build src and dst as bank + "_" + account
    df['src'] = df['from_bank'].astype(str) + '_' + df['src_acc'].astype(str)
    df['dst'] = df['to_bank'].astype(str) + '_' + df['dst_acc'].astype(str)
    
    # Add t = hours since the first transaction
    t_min = df['ts'].min()
    df['t'] = (df['ts'] - t_min).dt.total_seconds() / 3600.0
    
    # Sort by t
    df = df.sort_values('t').reset_index(drop=True)
    
    # Add eid as a row id
    df['eid'] = df.index.astype(int)
    
    # Normalise step: add amount_usd from static rate dictionary
    rate_dict = rates if rates is not None else STATIC_USD_RATES
    df['amount_usd'] = df['amount'] * df['cur_paid'].map(rate_dict).fillna(1.0)
    
    return df

def time_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by time into 60/20/20 train, val, and test splits.
    """
    if not df['t'].is_monotonic_increasing:
        df = df.sort_values('t').reset_index(drop=True)
    
    n = len(df)
    n_train = int(n * 0.60)
    n_val = int(n * 0.20)
    
    train = df.iloc[:n_train].copy().reset_index(drop=True)
    val = df.iloc[n_train:n_train + n_val].copy().reset_index(drop=True)
    test = df.iloc[n_train + n_val:].copy().reset_index(drop=True)
    
    return train, val, test

if __name__ == "__main__":
    print("Testing src/load.py on 200,000 rows slice...")
    df_slice = load("data/HI-Small_Trans.csv", nrows=200000)
    print("\nDistinct currencies in cur_paid:")
    currencies = df_slice['cur_paid'].unique().tolist()
    print(currencies)
    
    print("\nUSD Normalisation sample:")
    print(df_slice[['eid', 'cur_paid', 'amount', 'amount_usd']].head(10))
    
    train, val, test = time_split(df_slice)
    print("\nTime Split Shapes (60/20/20):")
    print(f"  Full slice shape: {df_slice.shape}")
    print(f"  Train shape:      {train.shape} (t: {train['t'].min():.2f}h to {train['t'].max():.2f}h)")
    print(f"  Val shape:        {val.shape} (t: {val['t'].min():.2f}h to {val['t'].max():.2f}h)")
    print(f"  Test shape:       {test.shape} (t: {test['t'].min():.2f}h to {test['t'].max():.2f}h)")
    
    laundering_count = (df_slice['label'] == 1).sum()
    laundering_rate = (df_slice['label'] == 1).mean()
    print(f"\nLaundering statistics on 200k slice:")
    print(f"  Illicit transactions: {laundering_count} / {len(df_slice)}")
    print(f"  Laundering rate:       {laundering_rate:.6f} ({laundering_rate * 100:.4f}%)")
