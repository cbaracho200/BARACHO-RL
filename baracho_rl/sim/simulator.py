from __future__ import annotations
from typing import Iterable, Dict, Any, List
import csv

class Simulator:
    """Stub de simulador: carrega logs e reproduz episódios; MVP para treinos offline."""
    @staticmethod
    def from_logs(path: str) -> List[Dict[str, Any]]:
        rows = []
        with open(path, "r") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                rows.append(r)
        return rows

    @staticmethod
    def synthetic_pricing_csv(path: str, months: int = 24):
        import math, random
        with open(path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["month","price","demand","revenue","cost","profit"])
            base_price, unit_cost = 10.0, 4.0
            for t in range(months):
                season = 1.0 + 0.2 * math.sin(2*math.pi*(t%12)/12)
                price = base_price * (0.9 + 0.2*random.random())
                demand = 1000 * season * (price/base_price) ** (-1.2) * (0.9 + 0.2*random.random())
                revenue = price*demand; cost = unit_cost*demand; profit = revenue - cost
                w.writerow([t, price, demand, revenue, cost, profit])
