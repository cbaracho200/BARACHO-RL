from __future__ import annotations
from typing import Dict, Any
from ..core.base import AgentBase

class RuleAgent(AgentBase):
    """Aumenta levemente o preço em meses pares e reduz em ímpares; exemplo didático."""
    def __init__(self, step: float = 0.5):
        self.step = step
    def act(self, obs: Dict[str, Any], ctx=None) -> Dict[str, float]:
        month = obs.get("month", 0)
        base = obs.get("baseline_price", 10.0)
        return {"price": base + (self.step if (month % 2 == 0) else -self.step)}
