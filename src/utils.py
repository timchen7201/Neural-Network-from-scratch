import numpy as np

def one_hot_encoding(y,class_num):
    y_one_hot = []
    for label in y:
        one_hot = np.zeros(class_num)
        one_hot[int(label)] = 1
        y_one_hot.append(one_hot)
    return np.asarray(y_one_hot)


def shuffle(X,y):
    if X.shape[0] != y.shape[0]:
        raise ValueError("The number of X and Y cannot match")
    
    permutation = np.random.permutation(X.shape[0])
    return X[permutation], y[permutation]

def get_batch(X,current_batch,batch_size,total_data_num):
    begin = current_batch* batch_size
    end = min(begin+batch_size,total_data_num)
    return X[begin:end]