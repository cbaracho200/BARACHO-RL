from __future__ import annotations
from typing import Iterable
import math

def npv(cashflows: Iterable[float], rate_annual: float, periods_per_year: int = 12) -> float:
    r = (1 + rate_annual) ** (1/periods_per_year) - 1
    return sum(cf / ((1 + r) ** t) for t, cf in enumerate(cashflows, start=1))

def irr(cashflows: Iterable[float], guess: float = 0.1, periods_per_year: int = 12, iters: int = 50) -> float:
    # Newton-Raphson simples (didático; não robusto para todos os casos)
    r = guess
    cfs = list(cashflows)
    for _ in range(iters):
        denom = [(1 + r) ** t for t in range(1, len(cfs)+1)]
        f = sum(cf/d for cf, d in zip(cfs, denom))
        df = sum(-t*cf/((1+r)**(t+1)) for t, cf in enumerate(cfs, start=1))
        if abs(df) < 1e-9: break
        r_new = r - f/df
        if abs(r_new - r) < 1e-9: break
        r = r_new
    return ((1 + r) ** periods_per_year) - 1

def cvar(returns: Iterable[float], alpha: float = 0.05) -> float:
    arr = sorted(returns)
    k = max(1, int(len(arr) * alpha))
    tail = arr[:k]
    return sum(tail)/len(tail)
