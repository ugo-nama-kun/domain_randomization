import gymnasium as gym
import domain_randomization

env = gym.make("DomainRandomization-v0", render_mode="human")

while True:
    done = False
    env.reset()
    while not done:
        obs, reward, terminal, truncated, info = env.step(env.action_space.sample())
        done = terminal | truncated
        env.render()
