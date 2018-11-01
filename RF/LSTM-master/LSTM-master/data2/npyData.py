import os
import numpy as np


def getData(path):
    pathDir = os.listdir(path)
    dataI = []
    dataQ = []
    labels = []
    count = 0
    for file in pathDir:
        count += 1
        print(count)
        filename = os.path.join(path, file)
        with open(filename, 'r') as fopen:
            data = np.loadtxt(fopen, delimiter=",", skiprows=0)
            fopen.close()
        dataI.append(data[:, 0])
        dataQ.append(data[:, 1])
        labels.append(int(file[8]))
    return dataI, dataQ, labels


if __name__ == "__main__":
    print("Loading Data")
    dataI, dataQ, labels = getData("AcquisitionData")

    print("Writing Data to npy")
    np.save("dataI.npy", dataI)
    np.save("dataQ.npy", dataQ)
    np.save("labels.npy", labels)
