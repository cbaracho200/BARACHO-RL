from __future__ import annotations
from typing import Dict, Any, Tuple
import math, random
from ..core.base import EnvBase

class DynamicPricingEnv(EnvBase):
    """Ambiente minimalista de precificação dinâmica com horizonte mensal.
    Estado: {month, baseline_price, demand_estimate}
    Ação: {price} (float)
    Recompensa: lucro = receita - custo variável
    """
    def __init__(self, horizon: int = 24, discount_rate_annual: float = 0.12,
                 base_demand: float = 1000.0, base_price: float = 10.0,
                 unit_cost: float = 4.0, elasticity: float = -1.2, seasonality: bool = True, rng_seed: int = 42):
        self.horizon = horizon
        self.discount_rate_annual = discount_rate_annual
        self.base_demand = base_demand
        self.base_price = base_price
        self.unit_cost = unit_cost
        self.elasticity = elasticity
        self.seasonality = seasonality
        self.rng = random.Random(rng_seed)
        self.reset()

    def reset(self) -> Dict[str, Any]:
        self.t = 0
        self._last_price = self.base_price
        return self._obs()

    def _season_factor(self, t: int) -> float:
        if not self.seasonality: return 1.0
        # sazonalidade simples senoidal
        return 1.0 + 0.2 * math.sin(2*math.pi * (t % 12) / 12)

    def _demand(self, price: float, t: int) -> float:
        season = self._season_factor(t)
        # demanda ~ (preço/baseline)^elasticity
        demand = self.base_demand * season * (price / self.base_price) ** (self.elasticity)
        # ruído
        demand *= (0.9 + 0.2 * self.rng.random())
        return max(0.0, demand)

    def _obs(self) -> Dict[str, Any]:
        return {
            "month": self.t,
            "baseline_price": self.base_price,
            "last_price": self._last_price,
            "demand_estimate": self._demand(self._last_price, self.t),
        }

    def step(self, action: Dict[str, float]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        price = float(action.get("price", self.base_price))
        price = max(0.01, price)
        d = self._demand(price, self.t)
        revenue = price * d
        cost = self.unit_cost * d
        profit = revenue - cost
        info = {"revenue": revenue, "cost": cost, "demand": d, "price": price}
        self._last_price = price
        self.t += 1
        done = self.t >= self.horizon
        return self._obs(), profit, done, info
