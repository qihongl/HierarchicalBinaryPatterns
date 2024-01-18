import numpy as np
import random
from copy import deepcopy
from utils import create_block_matrix, to_pth

class HierarchicalBinaryPatterns(object):

    def __init__(self, dim=32, n1=4, n2=4, n3=4, p1=.2, p2=.05):
        self.dim = dim
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.p1 = p1
        self.p2 = p2
        # useful constants
        self.n = n1 * n2 * n3
        self.relation_matrix = create_block_matrix(self.n, n1) + create_block_matrix(self.n, n1*n2) + np.eye(self.n)
        # gen data
        A = np.round(np.random.uniform(size=(n1, dim)))
        Aij = np.zeros((n1, n2, dim))
        Aijk = np.zeros((n1, n2, n3, dim))
        for i in range(n1):
            for j in range(n2):
                Aij[i, j] = random_flip(p1, deepcopy(A[i]))
                for k in range(n3):
                    Aijk[i,j,k] = random_flip(p2, deepcopy(Aij[i, j]))
        self.data = Aijk.reshape((-1, dim))

    def get_data(self):
        return to_pth(self.data)


def random_flip(p, vector):
    """
    Flips p percent of the entries in a binary-valued vector.

    :param p: float, percentage of entries to flip (between 0 and 100)
    :param vector: list or numpy array of binary values (0s and 1s)
    :return: modified vector with p percent of entries flipped
    """
    if not 0 <= p <= 1:
        raise ValueError("Percentage p must be between 0 and 100")

    num_to_flip = int(np.round(len(vector) * p))
    indices_to_flip = random.sample(range(len(vector)), num_to_flip)

    for index in indices_to_flip:
        vector[index] = 1 - vector[index]  # Flip the value (0->1, 1->0)

    return vector


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style='white', palette='colorblind', context='talk')

    '''generate patterns with 2 levels of Hierarchy
        4 level 1 nodes - prototypes
        4 leafs per prototypes
        4x4 = 16 instances in total
    '''
    dim = 256
    n1 = 4
    n2 = 1
    n3 = 4
    p1 = 0
    p2 = .1

    bup = HierarchicalBinaryPatterns(dim=dim, n1=n1,n2=n2,n3=n3, p1=p1, p2=p2)

    f, ax = plt.subplots()
    RSM = np.corrcoef(bup.data)
    sns.heatmap(RSM, square=True, cmap='RdYlBu_r', center=0)

    f, ax = plt.subplots()
    RSM = np.corrcoef(bup.data)
    sns.heatmap(RSM, square=True, cmap='RdYlBu_r', center=0)


    '''generate patterns with 3 levels of Hierarchy
        4 level 1 nodes - prototypes
        4 level 2 nodes per every prototype
        4 level 3 nodes / leafs per level 2 node
        4x4x4 = 64 instances in total
    '''
    dim = 256
    n1 = 4
    n2 = 4
    n3 = 4
    p1 = .1
    p2 = .1

    bup = HierarchicalBinaryPatterns(dim=dim, n1=n1,n2=n2,n3=n3, p1=p1, p2=p2)

    f, ax = plt.subplots()
    RSM = np.corrcoef(bup.data)
    sns.heatmap(RSM, square=True, cmap='RdYlBu_r', center=0)

    f, ax = plt.subplots()
    RSM = np.corrcoef(bup.data)
    sns.heatmap(RSM, square=True, cmap='RdYlBu_r', center=0)
