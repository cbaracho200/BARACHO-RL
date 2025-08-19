from __future__ import annotations
from typing import Dict, Any, List
from ..core.base import AgentBase

class SACAgent(AgentBase):
    """Stub didático de SAC; usa heurística simples para demonstração."""
    def __init__(self, policy: str = "GRU"):
        self.temp = 0.1
    def act(self, obs: Dict[str, Any], ctx=None) -> Dict[str, float]:
        p = float(obs.get("baseline_price", 10.0))
        last = float(obs.get("last_price", p))
        # ajusta levemente na direção do preço anterior (suavidade)
        price = 0.8*last + 0.2*p
        return {"price": price}
    def learn(self, batch: List[Dict[str, Any]]) -> None:
        pass
