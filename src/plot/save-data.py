import numpy as np

# set random seed
np.random.seed(99)

# generate data.
data = np.random.rand(10)*100
print(data)

# save data to text file
np.savetxt("data.csv", data, delimiter=",")