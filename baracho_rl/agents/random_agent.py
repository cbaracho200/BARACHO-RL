from __future__ import annotations
import random
from typing import Dict, Any
from ..core.base import AgentBase

class RandomAgent(AgentBase):
    def __init__(self, low: float = 5.0, high: float = 15.0, seed: int = 123):
        self.rng = random.Random(seed)
        self.low, self.high = low, high
    def act(self, obs: Dict[str, Any], ctx=None) -> Dict[str, float]:
        return {"price": self.rng.uniform(self.low, self.high)}
