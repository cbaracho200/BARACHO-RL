from __future__ import annotations
from typing import Dict, Any
from ..core.base import AgentBase

class OptionPolicy(AgentBase):
    """Macro-ação que fixa metas por k passos; stub didático."""
    def __init__(self, duration: int = 3):
        self.duration = duration
        self.counter = 0
        self.delta = 0.5
    def act(self, obs: Dict[str, Any], ctx=None) -> Dict[str, float]:
        if self.counter % self.duration == 0:
            # muda direção a cada option
            self.delta *= -1
        self.counter += 1
        base = float(obs.get("baseline_price", 10.0))
        return {"price": base + self.delta}

class HierarchicalAgent(AgentBase):
    def __init__(self, high: AgentBase, low: AgentBase):
        self.high, self.low = high, low
        self.last_goal = None
    def act(self, obs: Dict[str, Any], ctx=None) -> Dict[str, float]:
        # alto nível define deslocamento; baixo nível aplica ajuste fino
        goal = self.high.act(obs, ctx)
        low = self.low.act(obs, ctx)
        return {"price": 0.5*goal["price"] + 0.5*low["price"]}
