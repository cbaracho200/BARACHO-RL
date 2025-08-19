from __future__ import annotations
import json, os
import typer
from ..envs.registry import make_env
from ..core.trainer import Trainer
from ..algos.ppo import PPOAgent
from ..algos.sac import SACAgent
from ..algos.grpo import GRPOAgent

app = typer.Typer(help="BARACHO-RL CLI")

@app.command()
def train(env: str = "DynamicPricingEnv",
          algo: str = "PPO",
          horizon: int = 24,
          steps: int = 50_000,
          outdir: str = "runs/last"):
    e = make_env(env, horizon=horizon)
    agent = {"PPO": PPOAgent, "SAC": SACAgent, "GRPO": GRPOAgent}.get(algo, PPOAgent)()
    Trainer(e, agent, outdir=outdir).fit(steps=steps).report()

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

if __name__ == "__main__":
    app()
