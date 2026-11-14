from src.patterns import *
from src.patterns import parse_patterns as _raw_parse_patterns


def _norm_acc(a):
    bank, _, acc = a.partition("_")
    try:
        bank = str(int(bank))      # "020" -> "20", "00410" -> "410"
    except ValueError:
        pass
    return f"{bank}_{acc}"


def parse_patterns(path):
    rings = _raw_parse_patterns(path)
    for r in rings:
        r["accounts"] = {_norm_acc(a) for a in r["accounts"]}
    return rings