from __future__ import annotations
from typing import Dict, Any, Callable
import yaml, math
from ..envs.registry import make_env
from ..envs.compose import ComposeEnv

class _Proxy:
    def __init__(self, env, info):
        self._env = env; self._info = info
    def __getattr__(self, k):
        if isinstance(self._info, dict) and k in self._info:
            return self._info[k]
        return getattr(self._env, k)

def _eval_expr(expr: str, context: Dict[str, Any]) -> float:
    safe_builtins = {"__builtins__": {}}
    safe_funcs = {"min": min, "max": max, "abs": abs, "math": math}
    return eval(expr, {**safe_builtins, **safe_funcs}, context)

def build_from_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Simple non-compose
    if "compose" not in cfg:
        env_name = cfg.get("env", "DynamicPricingEnv")
        kwargs = cfg.get("env_cfg", {}) or {}
        env = make_env(env_name, **kwargs)
        algo = cfg.get("algo", "PPO")
        policy = cfg.get("policy", "MLP")
        return env, algo, {"policy": policy}
    # Compose
    comp = cfg["compose"]
    envs = {}
    for name, spec in comp["envs"].items():
        typ = spec.get("type")
        kwargs = {k: v for k, v in spec.items() if k != "type"}
        envs[name] = make_env(typ, **kwargs)
    weights = comp.get("weights", {})
    rules = comp.get("coupling", [])
    def coupler(envs_map, last_infos):
        # cria proxies por nome
        ctx = {name: _Proxy(envs_map[name], last_infos.get(name, {})) for name in envs_map}
        extra = 0.0
        for rule in rules:
            if "set" in rule:
                target = rule["set"]   # ex.: "Pricing.external_demand_mult"
                expr = rule["expr"]
                val = float(_eval_expr(expr, ctx))
                vmin, vmax = rule.get("clamp", [None, None])
                if vmin is not None: val = max(float(vmin), val)
                if vmax is not None: val = min(float(vmax), val)
                env_name, attr = target.split(".", 1)
                env_obj = envs_map[env_name]
                setter = f"set_{attr}"
                if hasattr(env_obj, setter):
                    getattr(env_obj, setter)(val)
                else:
                    setattr(env_obj, attr, val)
            elif "call" in rule:
                target = rule["call"]  # ex.: "Cash.apply_external_profit"
                args = [ _eval_expr(x, ctx) for x in rule.get("args", []) ]
                env_name, method = target.split(".", 1)
                getattr(envs_map[env_name], method)(*args)
            elif "reward" in rule:
                extra += float(_eval_expr(rule["reward"], ctx))
        if abs(extra) > 0:
            return {"reward": extra}
        return None
    env = ComposeEnv(envs, coupler=coupler, weights=weights)
    algo = cfg.get("algo", "PPO")
    policy = cfg.get("policy", "MLP")
    return env, algo, {"policy": policy}
