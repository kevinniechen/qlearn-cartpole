""" Brute force implementation for Cartpole """

import gym
import numpy as np

env = gym.make('CartPole-v0')

def run_episode(params):

	total_reward = 0

	observation = env.reset()
	while True:
		action = 0 if (params @ observation) < 0 else 1
		observation, reward, done, info = env.step(action)
		print(observation)

		total_reward += reward

		env.render()

		if done:
			break

	return total_reward


for i in range(100):
	params = np.random.rand(4)
	print("Policy Parameters: ", params)
	# params = [0.18507058, 0.43163198, 0.88900798, 0.98352915]  # policy
	total_reward = run_episode(params)
	print("Episode: ", i, " Reward: ", total_reward)
	if total_reward >= 200:
		print("Best Policy: ", params)
		break

env.close()
