from __future__ import annotations
from typing import Dict, Any, List
import csv, os, statistics, json
from .base import EnvBase, AgentBase, Transition
from .replay import ReplayBuffer
from .metrics import npv, irr, cvar

class Trainer:
    def __init__(self, env: EnvBase, agent: AgentBase, metrics: List[str] | None = None, outdir: str = "runs/last"):
        self.env, self.agent = env, agent
        self.buffer = ReplayBuffer()
        self.outdir = outdir
        self.metrics = metrics or ["NPV","IRR","CVaR"]
        os.makedirs(self.outdir, exist_ok=True)

    def fit(self, episodes: int | None = None, steps: int | None = None, discount_rate_annual: float = 0.12):
        assert (episodes or steps), "Defina episodes ou steps."
        ep = 0
        tot_steps = 0
        rewards_csv = os.path.join(self.outdir, "rewards.csv")
        with open(rewards_csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["episode","total_reward"])
            while True:
                obs = self.env.reset()
                done = False; total_reward = 0.0
                ep_transitions: List[Transition] = []
                while not done:
                    action = self.agent.act(obs, ctx={"t": getattr(self.env, "t", None)})
                    nxt, r, done, info = self.env.step(action)
                    tr = Transition(obs, action, r, nxt, done, info)
                    self.buffer.add(tr); ep_transitions.append(tr)
                    obs = nxt; total_reward += r; tot_steps += 1
                    if steps and tot_steps >= steps: done = True
                # exporta trajetória do episódio
                traj_path = os.path.join(self.outdir, f"trajectory_ep_{ep}.csv")
                with open(traj_path, "w", newline="") as tf:
                    tw = csv.writer(tf); tw.writerow(["t","obs","action","reward","done","info"])
                    for t, trn in enumerate(ep_transitions):
                        tw.writerow([t, trn.obs, trn.action, trn.reward, trn.done, trn.info])
                self.agent.learn([t.__dict__ for t in ep_transitions])  # batch learn stub
                w.writerow([ep, total_reward]); ep += 1
                if episodes and ep >= episodes: break
                if steps and tot_steps >= steps: break
        return self

    def report(self, discount_rate_annual: float = 0.12) -> Dict[str, Any]:
        rewards_path = os.path.join(self.outdir, "rewards.csv")
        episodes_rewards = []
        if os.path.exists(rewards_path):
            with open(rewards_path, "r") as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    episodes_rewards.append(float(row["total_reward"]))
        # KPIs
        kpis = {}
        if episodes_rewards:
            kpis["mean_reward"] = statistics.mean(episodes_rewards)
            kpis["NPV"] = npv(episodes_rewards, rate_annual=discount_rate_annual)
            kpis["IRR"] = irr(episodes_rewards) if len(episodes_rewards) > 2 else None
            kpis["CVaR5"] = cvar(episodes_rewards, alpha=0.05) if len(episodes_rewards) >= 20 else None
        with open(os.path.join(self.outdir, "report.json"), "w") as f:
            json.dump(kpis, f, indent=2)
        # plot simples de recompensa por episódio
        try:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(list(range(len(episodes_rewards))), episodes_rewards)
            plt.title("Total Reward por Episódio")
            plt.xlabel("Episódio"); plt.ylabel("Total Reward")
            png = os.path.join(self.outdir, "rewards.png")
            plt.savefig(png, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print("Plot falhou:", e)
        print("Report:", kpis)
        return kpis
