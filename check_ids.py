import pandas as pd
from src.patterns_fixed import parse_patterns

df = pd.read_pickle("data/cache.pkl")
rings = parse_patterns("data/HI-Small_Patterns.txt")
accs = set(df.src) | set(df.dst)

print("sample dataset accounts:", list(accs)[:5])
print("sample ring accounts   :", list(rings[0]["accounts"])[:5])
tot = sum(len(r["accounts"]) for r in rings)
hit = sum(len(r["accounts"] & accs) for r in rings)
print(f"ring accounts found in dataset: {hit} / {tot}")