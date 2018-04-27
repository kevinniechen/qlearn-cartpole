""" Visualizations for brute force implementation for Cartpole """

import gym
import numpy as np
import matplotlib.pyplot as plt


NUM_EPISODES=100  # max number of episodes before quitting to find a good policy
GOAL_REWARD=200  # reward necessary for a policy to be considered good
NUM_TRIALS=1000  # number of trials to test out how good this algorithm is


def run_episode(env, params):
    observation = env.reset()
    total_reward = 0
    while True:
        action = 0 if (params @ observation) < 0 else 1
        observation, reward, done, info = env.step(action)
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

def test_efficacy():
    # create graphs
    results = []
    for _ in range(NUM_TRIALS):
        results.append(train())
    plt.hist(results,50,normed=1, facecolor='g', alpha=0.8)
    plt.xlabel('Episodes required to reach' + str(GOAL_REWARD))
    plt.ylabel('Frequency')
    plt.title('Histogram of Random Search')
    plt.show()

    print("Average Episodes Needed For Good Policy: %s", np.sum(results) / float(NUM_TRIALS))

env = gym.make('CartPole-v0')
test_efficacy()