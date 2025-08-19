from __future__ import annotations
from typing import Any, Dict
from .dynamic_pricing import DynamicPricingEnv
from .hiring_capacity import HiringCapacityEnv
from .cash_management import CashManagementEnv
from .compose import ComposeEnv

_REGISTRY = {
    "DynamicPricingEnv": DynamicPricingEnv,
    "HiringCapacityEnv": HiringCapacityEnv,
    "CashManagementEnv": CashManagementEnv,
    "ComposeEnv": ComposeEnv,
}

def make_env(name: str, **kwargs) -> Any:
    if name not in _REGISTRY:
        raise KeyError(f"Env desconhecido: {name}. Disponíveis: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
