from __future__ import annotations
from typing import Dict, Any, List, Tuple, DefaultDict
from collections import defaultdict
import math, random
from ..core.base import AgentBase

class GRPOAgent(AgentBase):
    """Implementação didática do GRPO (sem crítico) para ações contínuas 1D.
    NOTA: Esta versão é educativa; não substitui uma implementação de produção.
    Atualiza um parâmetro 'mu' por grupo para deslocar a ação sugerida.
    """
    def __init__(self, group_key: str = "month_mod_3", lr: float = 0.01, entropy: float = 0.0):
        self.group_key = group_key
        self.lr = lr
        self.entropy = entropy
        self.mu: DefaultDict[Any, float] = defaultdict(lambda: 0.0)  # deslocamento
        self.baseline: DefaultDict[Tuple[Any, str], float] = defaultdict(lambda: 0.0)  # b[group, "price"]
        self.alpha = 0.2  # EMA
    def _group(self, obs: Dict[str, Any]) -> Any:
        m = int(obs.get("month", 0))
        return m % 3 if self.group_key == "month_mod_3" else "global"
    def act(self, obs: Dict[str, Any], ctx=None) -> Dict[str, float]:
        base = float(obs.get("baseline_price", 10.0))
        g = self._group(obs)
        noise = random.uniform(-0.5, 0.5) * self.entropy
        return {"price": max(0.01, base + self.mu[g] + noise)}
    def learn(self, batch: List[Dict[str, Any]]) -> None:
        # batch: list of dicts with keys [obs, action, reward, ...] conforme Trainer
        for tr in batch:
            obs = tr["obs"]; action = tr["action"]; r = float(tr["reward"])
            g = self._group(obs)
            a = float(action.get("price", obs.get("baseline_price", 10.0)))
            key = (g, "price")
            b = self.baseline[key]
            advantage = r - b
            # gradiente simples: move deslocamento na direção do sinal
            grad = math.copysign(1.0, advantage) * (a - float(obs.get("baseline_price", 10.0)))
            self.mu[g] += self.lr * grad
            # baseline EMA com clamp ascendente (não deixa diminuir demais)
            new_b = (1 - self.alpha) * b + self.alpha * r
            self.baseline[key] = max(b, new_b)
