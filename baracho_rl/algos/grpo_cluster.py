from __future__ import annotations
from typing import Dict, Any, List, Callable
from .grpo import GRPOAgent

class ClusterGRPOAgent:
    """Roteia para múltiplos GRPOAgents com base em uma função de cluster simples.
    - cluster_fn(obs) -> int em [0..n_clusters-1]
    - cada cluster tem sua própria baseline/parametrização (especialização)
    """
    def __init__(self, n_clusters: int = 3, cluster_fn: Callable[[Dict[str, Any]], int] | None = None, **kwargs):
        self.n_clusters = n_clusters
        self.cluster_fn = cluster_fn or (lambda obs: int(obs.get("month", 0)) % n_clusters)
        self.heads = [GRPOAgent(**kwargs) for _ in range(n_clusters)]
    def _idx(self, obs: Dict[str, Any]) -> int:
        try:
            k = int(self.cluster_fn(obs))
            return max(0, min(self.n_clusters-1, k))
        except Exception:
            return 0
    def act(self, obs: Dict[str, Any], ctx=None) -> Dict[str, float]:
        return self.heads[self._idx(obs)].act(obs, ctx)
    def learn(self, batch: List[Dict[str, Any]]) -> None:
        # separa por cluster e treina cada cabeça em seu segmento
        buckets = {i: [] for i in range(self.n_clusters)}
        for tr in batch:
            idx = self._idx(tr["obs"])
            buckets[idx].append(tr)
        for i in range(self.n_clusters):
            if buckets[i]:
                self.heads[i].learn(buckets[i])
    def save(self, path: str): pass
    def load(self, path: str): pass
