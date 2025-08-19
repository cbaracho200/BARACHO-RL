from baracho_rl import make_env, Trainer, SAC, RandomAgent, RuleAgent, GRPOAgent
env = make_env("DynamicPricingEnv", horizon=24)
agent = GRPOAgent(entropy=0.1)  # troque por SAC()/RuleAgent()/RandomAgent()
Trainer(env, agent).fit(episodes=50).report()
