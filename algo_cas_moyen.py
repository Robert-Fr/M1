from sklearn.datasets import load_iris
import random
import numpy as np


def algo_kmoyennes(precision,k,ens_x):
    c = []
    for i in range(0,k):
        c.append(ens_x[random.randint(0,len(ens_x))])
        print(c[i])
    l = 0
    while(True):
        for i in range(0,len(ens_x))


dataset = load_iris()
algo_kmoyennes(0,4,dataset.data)
print(len(dataset.data))