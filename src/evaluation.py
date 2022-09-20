import numpy as np


def accuracy(predict,y):
    return np.mean(np.argmax(predict,axis=1) == np.argmax(y,axis=1))

def cross_entropy(predict,y):
    # add small values to avoid log(0)
    nozero_output = np.add(predict, 1e-6)
    entropy = np.log(nozero_output)
    cross_entropy = -np.sum(np.multiply(y, entropy), axis=1, keepdims=True)

    # return mean entropy of this batch
    return np.mean(cross_entropy)
