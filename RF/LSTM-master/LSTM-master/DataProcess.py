import numpy as np
import matplotlib.pyplot as plt
import pdb


def loadData():
    dataI = np.load("data2/dataI.npy")
    dataQ = np.load("data2/dataQ.npy")
    labels = np.load("data2/labels.npy").astype(int)
    return dataI, dataQ, labels


def norm(data):
    data = data - np.amin(data, axis=1, keepdims=True)
    data = data / (np.amax(data, axis=1, keepdims=True) - np.amin(
        data, axis=1, keepdims=True))
    return data


def cutData(data, threshold):
    start = np.argmax(np.fabs(data) > threshold, axis=1)
    _data = np.zeros((start.shape[0], 6000), dtype=float)
    for i in range(start.shape[0]):
        _data[i] = data[i, start[i]:start[i] + 6000]
    return _data


def convertData(dataI, dataQ, labels, scale):

    # cut 6000 data points
    # 0.005 is the threshold
    dataI = cutData(dataI, 0.005)
    dataQ = cutData(dataQ, 0.005)

    print("Adding Noise")
    # plt.figure(1)
    # plt.subplot(211)
    # plt.plot(dataI[0,:])
    if scale > 0:
        addNoise(dataI, dataQ, scale)
    # plt.subplot(212)
    # plt.plot(dataI[0,:])
    # plt.show()

    # shuffle the data
    num_samples = labels.shape[0]
    assert num_samples == dataI.shape[0]
    perm = np.arange(num_samples)
    np.random.shuffle(perm)

    # one hot encode for labels
    one_hot = np.zeros((num_samples, np.max(labels)))
    one_hot[np.arange(num_samples), labels - 1] = 1

    # normalization
    dataI = norm(dataI)
    dataQ = norm(dataQ)

    # get the number for size of train set
    train = int(num_samples * 0.8)
    test = np.arange(train, num_samples)
    train = np.arange(train)
    train = perm[train]
    test = perm[test]

    # interweave the data
    Data = np.empty(
        (dataI.shape[0], dataI.shape[1] + dataQ.shape[1]), dtype=dataI.dtype)
    Data[:, 0::2] = dataI
    Data[:, 1::2] = dataQ

    trainLabel = one_hot[train]
    testLabel = one_hot[test]

    trainData = Data[train]
    testData = Data[test]

    return trainData, trainLabel, testData, testLabel


def addNoise(dataI, dataQ, scale):
    print("Scale", str(scale))

    for i in range(dataI.shape[1]):
        rmsI = np.sqrt(np.mean(np.square(dataI[i, :]))) * scale
        rmsQ = np.sqrt(np.mean(np.square(dataQ[i, :]))) * scale
        noiseI = np.random.normal(0, rmsI, dataI.shape[1])
        noiseQ = np.random.normal(0, rmsQ, dataQ.shape[1])
        dataI[i, :] += noiseI
        dataQ[i, :] += noiseQ

    return dataI, dataQ
