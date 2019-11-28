import numpy as np

x = np.array([-1, 1])
print(np.where(x>0))

n_classes = 5
def one_hot(y):
    ret = np.zeros((len(y), n_classes))
    ret[np.arange(len(y)),y] = 1
    return ret

print(one_hot(np.array([4,1,2,3,4])))


def tanh(x, grad=False):
    if grad:
        return 1 - tanh(x) ** 2

    isnt_arr = False
    try:
        temp = x.shape
    except:
        isnt_arr = True

    if isnt_arr:
        x = np.array([x])

    # x = np.tanh(x)

    e2x = np.exp(2 * x)
    x = (e2x - 1) / (e2x + 1)

    if isnt_arr:
        return x[0]
    else:
        return x

print(tanh(np.array([[0,1,3,5,-1],[0,1,3,5,-1]])))