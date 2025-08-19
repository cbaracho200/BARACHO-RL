from __future__ import annotations
import json, os
import typer
from typing import Dict, Any
from ..envs.registry import make_env
from ..envs.compose import ComposeEnv
from ..envs.dynamic_pricing import DynamicPricingEnv
from ..envs.hiring_capacity import HiringCapacityEnv
from ..envs.cash_management import CashManagementEnv
from ..core.trainer import Trainer
from ..algos.ppo import PPOAgent
from ..algos.sac import SACAgent
from ..algos.grpo import GRPOAgent
from ..refiner.refiner import Refiner

app = typer.Typer(help="BARACHO-RL CLI")

@app.command()
def train(env: str = "DynamicPricingEnv",
          algo: str = "PPO",
          horizon: int = 24,
          steps: int = 50_000,
          outdir: str = "runs/last"):
    if env == "ComposeDemo":
        # composição pronta: Hiring -> Pricing (demand_mult), Pricing profit -> Cash
        pr = DynamicPricingEnv(horizon=horizon)
        hi = HiringCapacityEnv(horizon=horizon)
        ca = CashManagementEnv(horizon=horizon)
        def coupler(envs: Dict[str, Any], infos: Dict[str, Dict[str, Any]]):
            cap = infos.get("Hiring", {}).get("capacity", hi.capacity if hasattr(hi, "capacity") else 100.0)
            base = hi.base_capacity
            mult = max(0.2, min(2.0, 0.5 + cap/(base+1e-6)))
            envs["Pricing"].set_external_demand_mult(mult)
            profit = infos.get("Pricing", {}).get("revenue", 0.0) - infos.get("Pricing", {}).get("cost", 0.0)
            envs["Cash"].apply_external_profit(profit)
        env_obj = ComposeEnv({"Hiring": hi, "Pricing": pr, "Cash": ca}, coupler=coupler)
    else:
        env_obj = make_env(env, horizon=horizon)
    agent = {"PPO": PPOAgent, "SAC": SACAgent, "GRPO": GRPOAgent}.get(algo, PPOAgent)()
    Trainer(env_obj, agent, outdir=outdir).fit(steps=steps).report()

@app.command()
def report(path: str = "runs/last/report.json"):
    if os.path.exists(path):
        print(open(path).read())
    else:
        print("Relatório não encontrado:", path)

@app.command()
def simulate(months: int = 24, out: str = "simulated.csv"):
    from ..sim.simulator import Simulator
    Simulator.synthetic_pricing_csv(out, months=months)
    print("Gerado:", out)

@app.command()
def refine(env: str = "DynamicPricingEnv",
           algo: str = "GRPO",
           horizon: int = 24,
           episodes: int = 40,
           window: int = 3,
           refine_episodes: int = 20,
           outdir: str = "runs/refined"):
    # cria fábrica de ambientes para o refiner
    def factory():
        if env == "ComposeDemo":
            pr = DynamicPricingEnv(horizon=horizon)
            hi = HiringCapacityEnv(horizon=horizon)
            ca = CashManagementEnv(horizon=horizon)
            def coupler(envs, infos):
                cap = infos.get("Hiring", {}).get("capacity", hi.capacity if hasattr(hi, "capacity") else 100.0)
                base = hi.base_capacity
                mult = max(0.2, min(2.0, 0.5 + cap/(base+1e-6)))
                envs["Pricing"].set_external_demand_mult(mult)
                profit = infos.get("Pricing", {}).get("revenue", 0.0) - infos.get("Pricing", {}).get("cost", 0.0)
                envs["Cash"].apply_external_profit(profit)
            return ComposeEnv({"Hiring": hi, "Pricing": pr, "Cash": ca}, coupler=coupler)
        return make_env(env, horizon=horizon)
    agent = {"PPO": PPOAgent, "SAC": SACAgent, "GRPO": GRPOAgent}.get(algo, GRPOAgent)()
    r = Refiner(window=window, refine_episodes=refine_episodes).refine(factory, agent, episodes=episodes)
    print("RIL:", r)
    # treino pós-refino
    Trainer(factory(), agent, outdir=outdir).fit(episodes=episodes).report()

if __name__ == "__main__":
    app()
