from __future__ import annotations
from typing import Any, Dict
from .dynamic_pricing import DynamicPricingEnv

_REGISTRY = {
    "DynamicPricingEnv": DynamicPricingEnv,
}

def make_env(name: str, **kwargs) -> Any:
    if name not in _REGISTRY:
        raise KeyError(f"Env desconhecido: {name}. Disponíveis: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
