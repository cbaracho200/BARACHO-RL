from __future__ import annotations
from typing import Dict, Any, Tuple
from ..core.base import EnvBase

class HiringCapacityEnv(EnvBase):
    """Ambiente simples de capacidade por contratação.
    Estado: {month, capacity}
    Ação: {hire} (int/float, variação da capacidade)
    Recompensa: -custo_fixo - custo_var_por_capacidade
    Info: {'capacity': capacidade_atual}
    """
    def __init__(self, horizon: int = 24, base_capacity: float = 100.0,
                 cost_per_cap: float = 1.0, fixed_cost: float = 50.0, rng_seed: int = 123):
        self.horizon = horizon
        self.base_capacity = base_capacity
        self.capacity = base_capacity
        self.cost_per_cap = cost_per_cap
        self.fixed_cost = fixed_cost
        self.rng_seed = rng_seed

    def reset(self) -> Dict[str, Any]:
        self.t = 0
        self.capacity = self.base_capacity
        return {"month": self.t, "capacity": self.capacity}

    def step(self, action: Dict[str, float]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        hire = float(action.get("hire", 0.0))
        self.capacity = max(0.0, self.capacity + hire)
        # custo mensal: fixo + proporcional à capacidade
        reward = -(self.fixed_cost + self.cost_per_cap * self.capacity)
        info = {"capacity": self.capacity, "hire": hire}
        self.t += 1
        done = self.t >= self.horizon
        obs = {"month": self.t, "capacity": self.capacity}
        return obs, reward, done, info
