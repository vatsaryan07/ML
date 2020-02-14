import math as m
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
sn.set()
points = np.random.rand(100,2)*100          # assigning the train array random variables within limits of 20
out = np.random.randint(0,4,size=100)      # assigning values to the training set from 0 to 3
col = {0:'red',1:'blue',2:'yellow',3:'green'}    # assigning colours to the classes as given
for x in range(len(out)):
  plt.scatter(points[x][0],points[x][1],c = col[out[x]])        # plotting the training set
plt.title("Existing points")
plt.show()
p = np.array([0,0])
print("Enter the points")
p[0] = float(input())
p[1] = float(input())
for x in range(len(out)):
  plt.scatter(points[x][0],points[x][1],c = col[out[x]])
plt.scatter(p[0],p[1],c = 'black',marker='*',s= 500)             # highlighting the input point amongst the training set
plt.title("New Point")
plt.show()
out.reshape(-1)
kclass = KNeighborsClassifier(n_neighbors=5)                     # taking 5 nearest neighbours to avoid overfitting
kclass.fit(points,out)
li = kclass.predict([p])
kk = li[0]
print("The class of the input value point is ",col[kk])
points = np.append(points,[p],axis=0)
out = np.append(out,[kk],axis=0)
for x in range(len(out)):
  plt.scatter(points[x][0],points[x][1],c = col[out[x]])
plt.show()