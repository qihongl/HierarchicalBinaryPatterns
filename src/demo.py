from task.HierarchicalBinaryPatterns import HierarchicalBinaryPatterns
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

hbp = HierarchicalBinaryPatterns(dim=dim, n1=n1,n2=n2,n3=n3, p1=p1, p2=p2)

f, ax = plt.subplots()
RSM = np.corrcoef(hbp.data)
sns.heatmap(RSM, square=True, cmap='RdYlBu_r', center=0)

f, ax = plt.subplots()
RSM = np.corrcoef(hbp.data)
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

hbp = HierarchicalBinaryPatterns(dim=dim, n1=n1,n2=n2,n3=n3, p1=p1, p2=p2)

f, ax = plt.subplots()
RSM = np.corrcoef(hbp.data)
sns.heatmap(RSM, square=True, cmap='RdYlBu_r', center=0)

f, ax = plt.subplots()
RSM = np.corrcoef(hbp.data)
sns.heatmap(RSM, square=True, cmap='RdYlBu_r', center=0)
