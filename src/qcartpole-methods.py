""" Q-Learning methods and tests for Cartpole """

import gym
import numpy as np
import collections
import math
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

# hyperparameters
buckets=(1, 1, 6, 12,)
n_episodes=1000
goal_duration=195
min_alpha=0.1  # learning rate
min_epsilon=0.1  # exploration rate
gamma=1.0  # discount factor
ada_divisor=25
Q = np.zeros(buckets + (env.action_space.n,))

# helper functions
def discretize(obs):
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]
    ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
    new_obs = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
    new_obs = [min(buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
    return tuple(new_obs)

def choose_action(state, epsilon):
    return env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(Q[state])

def update_q(state_old, action, reward, state_new, alpha):
    Q[state_old][action] += alpha * (reward + gamma * np.max(Q[state_new]) - Q[state_old][action])

def get_epsilon(t):
    return max(min_epsilon, min(1, 1.0 - math.log10((t + 1) / ada_divisor)))

def get_alpha(t):
    return max(min_alpha, min(1.0, 1.0 - math.log10((t + 1) / ada_divisor)))

# tests
def test_discretize():
    observation = env.reset()
    print("Continuous Observation: ", observation)
    state = discretize(observation)
    print("Discretized Observation: ", state)

def test_q():
    state = discretize(env.reset())
    print("Q Table: ", Q)
    print("Q Table Dimensions: ", Q.shape)
    print("Discretized Observation: ", state)
    print("Q Value at State: ", Q[state])
    Q[state] = 999
    print("Updated Q Table: ", Q)

def plot_epsilon():
    results = []
    for i in range(1000):
        results.append(get_epsilon(i))
    plt.plot(results)
    plt.xlabel('Time')
    plt.ylabel('Epsilon (Exploration Rate)')
    plt.title('Epsilon Curve')
    plt.show()

def plot_alpha():
    results = []
    for i in range(1000):
        results.append(get_alpha(i))
    plt.plot(results)
    plt.xlabel('Time')
    plt.ylabel('Alpha (Learning Rate)')
    plt.title('Alpha Curve')
    plt.show()

if __name__ == '__main__':
    test_discretize()
    # test_q()
    # plot_epsilon()
    # plot_alpha()
