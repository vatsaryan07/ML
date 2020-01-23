import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from numpy.linalg import norm as normal
style.use('ggplot')

class KM:
    def __init__(self,k=2,tol=0.001,max = 3000):
        self.k = k
        self.tol = tol
        self.max = max

    def fit(self,x):
        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] = x[i]

        for i in range(self.max):
            self.classes = {}
            for i in range(self.k):
                self.classes[i] = []
            for i in x:
                dist = [normal(i - self.centroids[l]) for l in self.centroids]                 # Using the normal function to calculate the Frobeius norm of the two points ie Difference of sum of squares
                ck = dist.index(min(dist))                                                     # Finding the smallest of the centroids
                self.classes[ck].append(i)
            prev = dict(self.centroids)
            for i in self.classes:
                self.centroids[i] = np.average(self.classes[i],axis=0)                         # Updating the centroid
            check = True
            for i in self.centroids:
                oc = prev[i]
                cc = self.centroids[i]
                if np.sum((oc-cc)/oc*100)<self.tol:
                    check = False
            if check:
                break

    def cent(self):
        return self.centroids

    def pred(self,data):
        dist = [normal(data - self.centroids[l]) for l in self.centroids]
        ck = dist.index(min(dist))
        return ck
Y = np.random.randint(low=0,high=20,size=(10,2))
X = [(1,2),(2,3),(10,19),(13,17),(4,1),(5,10)]
reg = KM()
reg.fit(Y)
clr = ['g','r','y','b','o','v']
#for k in reg.centroids:
    #plt.scatter(reg.centroids[k][0],reg.centroids[k][1],c='k')
for k in reg.classes:
    print(reg.classes[k][0][1])
    colour = clr[k]
    for l in reg.classes[k]:
        print(l)
        plt.scatter(l[0],l[1],c=colour)

x1 = [10,20]
print("Enter the point to be classified\n")
x1[0] = float(input())
x1[1] = float(input())
plt.scatter(x1[0],x1[1],marker='*',s = 500)
plt.show()
l = reg.pred(x1)
print(np.array(X))
for k in reg.classes:
    colour = clr[k]
    for x in reg.classes[k]:
        plt.scatter(x[0],x[1],c=colour)
plt.scatter(x1[0],x1[1],c=clr[l])
plt.show()
