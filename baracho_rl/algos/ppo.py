from __future__ import annotations
from typing import Dict, Any, List
from ..core.base import AgentBase

class PPOAgent(AgentBase):
    """Stub didático de PPO; aqui apenas segue uma regra simples.
    Substitua por implementação real quando integrar backends.
    """
    def __init__(self):
        pass
    def act(self, obs: Dict[str, Any], ctx=None) -> Dict[str, float]:
        # aproxima comportamento: microajusta com base em demanda estimada
        p = float(obs.get("baseline_price", 10.0))
        d = float(obs.get("demand_estimate", 1000.0))
        return {"price": p * (1.0 + (0.0001*(d-1000)))}
    def learn(self, batch: List[Dict[str, Any]]) -> None:
        pass
