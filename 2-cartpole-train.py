import gym
import numpy as np


NUM_EPISODES=100  # max number of episodes before quitting to find a good policy
GOAL_REWARD=200  # reward necessary for a policy to be considered good


def run_episode(env, params):
    observation = env.reset()
    total_reward = 0
    while True:
        action = 0 if (params @ observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        env.render()
        total_reward += reward
        if done:
            break
    return total_reward

def train():
    best_params = None
    best_reward = 0
    for counter in range(NUM_EPISODES):
        params = np.random.rand(4) * 2 - 1
        reward = run_episode(env, params)
        if reward > best_reward:
            best_reward = reward
            best_params = params
            if reward == GOAL_REWARD:
                break

    return counter

env = gym.make('CartPole-v0')
train()