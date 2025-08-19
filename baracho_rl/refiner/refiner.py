from __future__ import annotations
from typing import Dict, Any, List, Tuple, Callable
import math
from ..core.base import EnvBase, AgentBase, Transition

class Refiner:
    """Recursive Improvement Loop (RIL)
    - executa episódios, identifica a pior janela (rolling), e faz fine-tune local
    """
    def __init__(self, window: int = 3, refine_episodes: int = 20):
        self.window = window
        self.refine_episodes = refine_episodes

    def _run_episode(self, env: EnvBase, agent: AgentBase) -> Tuple[List[Transition], float]:
        obs = env.reset()
        done = False
        traj: List[Transition] = []
        total = 0.0
        while not done:
            act = agent.act(obs, ctx={"t": getattr(env, "t", None)})
            nxt, r, done, info = env.step(act)
            traj.append(Transition(obs, act, r, nxt, done, info))
            obs = nxt; total += r
        return traj, total

    def _worst_window(self, traj: List[Transition]) -> Tuple[int, float]:
        # retorna início da pior janela por soma de recompensas
        rewards = [tr.reward for tr in traj]
        n = len(rewards)
        w = self.window
        worst_sum = math.inf; worst_i = 0
        for i in range(0, max(1, n - w + 1)):
            s = sum(rewards[i:i+w])
            if s < worst_sum:
                worst_sum, worst_i = s, i
        return worst_i, worst_sum

    def refine(self, env_factory: Callable[[], EnvBase], agent: AgentBase, episodes: int = 20) -> Dict[str, Any]:
        # 1) coleta traj e acha janela ruim
        traj, total = self._run_episode(env_factory(), agent)
        start, score = self._worst_window(traj)
        # 2) fine-tune: recomeça na janela ruim via "queima" de passos com política atual
        for _ in range(self.refine_episodes):
            env = env_factory()
            # fast-forward até 'start'
            obs = env.reset()
            for i in range(start):
                a = agent.act(obs, ctx={"t": getattr(env, "t", None)})
                obs, _, done, _ = env.step(a)
                if done: break
            # aprende apenas na janela
            steps = 0
            local_batch = []
            while steps < self.window:
                a = agent.act(obs, ctx={"t": getattr(env, "t", None)})
                nxt, r, done, info = env.step(a)
                local_batch.append({"obs": obs, "action": a, "reward": r, "next": nxt, "done": done, "info": info})
                obs = nxt; steps += 1
                if done: break
            # passa batch para learn()
            agent.learn(local_batch)
        return {"start": start, "score": score}
