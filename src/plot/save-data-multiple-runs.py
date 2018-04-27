import numpy as np

for i in range(5):
    # set random seed
    np.random.seed(i)

    # generate data.
    data = np.random.rand(10)*100
    print(data)

    # save data to text files
    DATAPATH = "data/"
    np.savetxt("%sdata-run-%s.csv" % (DATAPATH, i), data, delimiter=",")