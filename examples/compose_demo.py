from baracho_rl.cli.main import train
# roda composição Hiring -> Pricing -> Cash com GRPO
train(env="ComposeDemo", algo="GRPO", horizon=24, steps=20000, outdir="runs/compose")
