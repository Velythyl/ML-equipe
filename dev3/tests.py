import numpy as np

x = np.array([-1, 1])
print(np.where(x>0))

n_classes = 5
def one_hot(y):
    ret = np.zeros((len(y), n_classes))
    ret[np.arange(len(y)),y] = 1
    return ret

print(one_hot(np.array([4,1,2,3,4])))