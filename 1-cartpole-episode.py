import gym
import numpy as np


env = gym.make('CartPole-v0')  # setup OpenAI cartpole env

params = np.random.rand(4)  # parameters for random policy

print("Policy: ", params)

# run an episode
observation = env.reset()
total_reward = 0

while True:
    action = 0 if (params @ observation) < 0 else 1  # policy
    observation, reward, done, info = env.step(action)
    env.render()
    total_reward += reward
    if done:
        break

print("Reward: ", total_reward)