from __future__ import annotations
from typing import Dict, Any, Tuple, Callable, OrderedDict
from collections import OrderedDict as OD
from ..core.base import EnvBase

class ComposeEnv(EnvBase):
    """Composição de múltiplos ambientes com acoplamentos via 'coupler'.
    - envs: dict name->EnvBase
    - weights: dict name->float (pesa recompensa de cada subambiente)
    - coupler: callable(envs, last_infos) -> Optional[dict] (pode ajustar envs e retornar bônus/penalidade)
      Ex.: return {"reward_bonus": +x} ou {"reward_penalty": -y} ou {"reward": x}
    Ação: dict name->subaction; Obs/Info: dict name->sub{obs,info}; Recompensa: soma ponderada + ajuste
    """
    def __init__(self, envs: Dict[str, EnvBase], coupler: Callable[[Dict[str, EnvBase], Dict[str, Dict[str, Any]]], dict | None] | None = None, weights: Dict[str, float] | None = None):
        self.envs = OD(envs)
        self.coupler = coupler
        self.weights = weights or {k: 1.0 for k in self.envs}
        self._last_infos: Dict[str, Dict[str, Any]] = {}
        self.horizon = min(getattr(e, "horizon", 10**9) for e in self.envs.values())

    def reset(self) -> Dict[str, Any]:
        obs = {}
        for k, e in self.envs.items():
            obs[k] = e.reset()
        self._last_infos = {k: {} for k in self.envs}
        return obs

    def step(self, action: Dict[str, Dict[str, float]]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        adjust = None
        if self.coupler:
            adjust = self.coupler(self.envs, self._last_infos)
        total_reward = 0.0
        next_obs, info = {}, {}
        done_flags = []
        for name, env in self.envs.items():
            subact = action.get(name, {})
            nobs, r, done, inf = env.step(subact)
            next_obs[name] = nobs
            info[name] = inf
            total_reward += self.weights.get(name, 1.0) * r
            done_flags.append(done)
        if isinstance(adjust, dict):
            if "reward" in adjust: total_reward += float(adjust["reward"])
            total_reward += float(adjust.get("reward_bonus", 0.0))
            total_reward += float(adjust.get("reward_penalty", 0.0))  # tipicamente negativo
        self._last_infos = info
        done = any(done_flags)
        return next_obs, total_reward, done, info
