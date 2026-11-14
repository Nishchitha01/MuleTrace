import datetime
import sys
import time
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def log(msg: str) -> None:
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now_str}] {msg}", flush=True)

def main():
    log("Starting diag.py")
    
    # Import modules
    log("Importing src.load and src.detectors...")
    from src.load import load, time_split
    from src.detectors import fan, chains, temporal_cycles
    log("Imports successful.")

    # Load FULL dataset
    data_path = str(ROOT / "data" / "HI-Small_Trans.csv")
    log(f"Loading FULL dataset from {data_path}...")
    t0 = time.time()
    df = load(data_path)
    load_time = time.time() - t0
    log(f"Full dataset loaded in {load_time:.2f}s.")

    # Total rows and illicit count
    total_rows = len(df)
    illicit_total = int(df["label"].sum())
    log(f"Total rows: {total_rows:,} | Illicit count: {illicit_total:,} ({illicit_total / total_rows * 100:.4f}%)")

    # Time split (60/20/20)
    log("Splitting data into train/val/test (60/20/20) by time...")
    train, val, test = time_split(df)
    
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        n_rows = len(split)
        n_illicit = int(split["label"].sum())
        t_min = split["t"].min()
        t_max = split["t"].max()
        pct = n_illicit / n_rows * 100 if n_rows > 0 else 0.0
        log(f"{name} split: {n_rows:,} rows | Illicit: {n_illicit:,} ({pct:.4f}%) | t range: [{t_min:.2f}h, {t_max:.2f}h] (span: {t_max - t_min:.2f}h)")

    # 1-day window of the test split
    log("Selecting 1-day (24h) window of the test split...")
    t_test_min = test["t"].min()
    t_test_window_end = t_test_min + 24.0
    test_1day = test[(test["t"] >= t_test_min) & (test["t"] <= t_test_window_end)].copy().reset_index(drop=True)
    n_1day = len(test_1day)
    n_illicit_1day = int(test_1day["label"].sum())
    log(f"Test 1-day window: {n_1day:,} rows | Illicit: {n_illicit_1day:,} | t range: [{test_1day['t'].min():.2f}h, {test_1day['t'].max():.2f}h]")

    # Run detectors separately on the 1-day test window
    # 1. fan-out (key="src", other="dst")
    log("Running fan-out detector (key='src', other='dst')...")
    t0 = time.time()
    try:
        fo_hits = fan(test_1day, key="src", other="dst", k=8, W=48.0)
        dt = time.time() - t0
        log(f"fan-out finished in {dt:.2f}s | Hits: {len(fo_hits)}")
    except Exception as e:
        dt = time.time() - t0
        log(f"fan-out FAILED after {dt:.2f}s with exception: {type(e).__name__}: {e}")

    # 2. fan-in (key="dst", other="src")
    log("Running fan-in detector (key='dst', other='src')...")
    t0 = time.time()
    try:
        fi_hits = fan(test_1day, key="dst", other="src", k=8, W=48.0)
        dt = time.time() - t0
        log(f"fan-in finished in {dt:.2f}s | Hits: {len(fi_hits)}")
    except Exception as e:
        dt = time.time() - t0
        log(f"fan-in FAILED after {dt:.2f}s with exception: {type(e).__name__}: {e}")

    # 3. chains
    log("Running chains detector (D=6.0, keep=0.70)...")
    t0 = time.time()
    try:
        chain_hits = chains(test_1day, D=6.0, keep=0.70)
        dt = time.time() - t0
        log(f"chains finished in {dt:.2f}s | Hits: {len(chain_hits)}")
    except Exception as e:
        dt = time.time() - t0
        log(f"chains FAILED after {dt:.2f}s with exception: {type(e).__name__}: {e}")

    # 4. temporal_cycles
    log("Running temporal_cycles detector (max_len=5, W=168.0)...")
    t0 = time.time()
    try:
        cycle_hits = temporal_cycles(test_1day, max_len=5, W=168.0)
        dt = time.time() - t0
        log(f"temporal_cycles finished in {dt:.2f}s | Hits: {len(cycle_hits)}")
    except Exception as e:
        dt = time.time() - t0
        log(f"temporal_cycles FAILED after {dt:.2f}s with exception: {type(e).__name__}: {e}")

    log("diag.py completed successfully.")

if __name__ == "__main__":
    main()
