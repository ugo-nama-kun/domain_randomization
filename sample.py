import gymnasium as gym
import domain_randomization

env = gym.make("DomainRandomization-v0")

while True:
    done = False
    obs, info = env.reset()
    while not done:
        print(obs, info)
        obs, reward, terminal, truncated, info = env.step(env.action_space.sample())
        done = terminal | truncated
        # env.render()
