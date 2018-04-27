import numpy as np
import matplotlib.pyplot as plt

# load data from text file
data = np.genfromtxt('data.csv', delimiter=',')

# plot data
plt.plot(data)
plt.xlabel('Reward')
plt.ylabel('Episode')
plt.title('Reward Plot for Fakelearn')
plt.show()