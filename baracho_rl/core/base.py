from __future__ import annotations
from typing import Any, Dict, Tuple, Optional, List
from dataclasses import dataclass

class EnvBase:
    """Interface mínima de ambiente (Gym-like)."""
    def reset(self) -> Dict[str, Any]:
        raise NotImplementedError
    def step(self, action: Dict[str, float]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        raise NotImplementedError

class AgentBase:
    """Agente minimalista: implemente act() e opcionalmente learn()."""
    def act(self, obs: Dict[str, Any], ctx: Dict[str, Any] | None = None) -> Dict[str, float]:
        raise NotImplementedError
    def learn(self, batch: List[Dict[str, Any]]) -> None:
        pass
    def save(self, path: str) -> None:
        pass
    def load(self, path: str) -> None:
        pass

class PolicyBase:
    """Placeholder para políticas baseadas em redes; não implementado neste MVP."""
    pass

@dataclass
class Transition:
    obs: Dict[str, Any]
    action: Dict[str, float]
    reward: float
    next_obs: Dict[str, Any]
    done: bool
    info: Dict[str, Any]
