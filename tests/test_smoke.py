from baracho_rl import make_env, Trainer, RuleAgent

def test_smoke():
    env = make_env("DynamicPricingEnv", horizon=12)
    agent = RuleAgent()
    tr = Trainer(env, agent, outdir="runs/test")
    tr.fit(episodes=3).report()
    assert True
