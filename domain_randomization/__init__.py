from gymnasium.envs.registration import register

register(
    id='DomainRandomization-v0',
    entry_point='domain_randomization.envs:BallEnv',
    max_episode_steps=100,
)
