import numpy as np
import matplotlib.pyplot as plt

DATAPATH = "data/"
files = ['data-run-0.csv', 'data-run-1.csv', 'data-run-2.csv', 'data-run-3.csv', 'data-run-4.csv']

# load data from text file
data = [0] * len(files)
for i, file in enumerate(files):
    data[i] = np.genfromtxt("%s%s" % (DATAPATH, file), delimiter=',')

# get average line
average = np.mean(data, axis=0)

# plot data
for i, file in enumerate(files):
    plt.plot(data[i], label=file)
plt.plot(average, label='Average', linewidth=8)
plt.xlabel('Reward')
plt.ylabel('Episode')
plt.title('Reward Plot for Fakelearn')
plt.legend()
plt.show()