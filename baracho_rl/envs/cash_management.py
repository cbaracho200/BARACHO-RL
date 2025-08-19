from __future__ import annotations
from typing import Dict, Any, Tuple
from ..core.base import EnvBase

class CashManagementEnv(EnvBase):
    """Gerencia caixa com juros simples; recebe lucros externos da composição.
    Estado: {month, cash}
    Ação: {invest} (0..1 proporção em ativo com rendimento r)
    Recompensa: variação do caixa (entrada externa + juros - custo)
    """
    def __init__(self, horizon: int = 24, initial_cash: float = 10000.0, rate_monthly: float = 0.01, admin_cost: float = 20.0):
        self.horizon = horizon
        self.initial_cash = initial_cash
        self.rate_monthly = rate_monthly
        self.admin_cost = admin_cost
        self.external_profit = 0.0

    def apply_external_profit(self, profit: float):
        self.external_profit += float(profit)

    def reset(self) -> Dict[str, Any]:
        self.t = 0
        self.cash = self.initial_cash
        self.external_profit = 0.0
        return {"month": self.t, "cash": self.cash}

    def step(self, action: Dict[str, float]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        invest = max(0.0, min(1.0, float(action.get("invest", 0.5))))
        # juros sobre parte investida
        interest = (self.cash * invest) * self.rate_monthly
        # aplica lucro externo acumulado (ex.: da precificação)
        self.cash += self.external_profit + interest - self.admin_cost
        reward = self.external_profit + interest - self.admin_cost
        info = {"invest": invest, "interest": interest, "external_profit": self.external_profit, "cash": self.cash}
        self.external_profit = 0.0  # consome o lucro aplicado
        self.t += 1
        done = self.t >= self.horizon
        obs = {"month": self.t, "cash": self.cash}
        return obs, reward, done, info
