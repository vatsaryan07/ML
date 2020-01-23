import math as m
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
sn.set()
def classify(group,p,k):

    distances = []
    for x in group:
        for y in group[x]:
            dist = m.sqrt((y[0] - p[0])**2 + (y[1] - p[1])**2)                           # Taking the Euclidean distance between the two points for all points in the dataset
            distances.append((dist,x))                                                   # Appending the distance and the group the point belongs to to the distance list
    distances = sorted(distances)[:k]                                                    # Sorting the distances and taking the nearest k neighbours
    f1 = 0
    f2 = 0
    for d in distances:
        if d[1]==0:
            f1= f1+1
        else:
            f2 = f2+1
    if f1>f2:
        return 0                                                                         # Depending on the majority of Neighbours the point gets classified
    else:
        return 1


#points = {0: [(1, 12), (2, 5), (3, 6), (3, 10), (3.5, 8), (2, 11), (2, 9), (1, 7)],
 #         1: [(5, 3), (3, 2), (1.5, 9), (7, 2), (6, 1), (3.8, 1), (5.6, 4), (4, 2), (2, 5)]}
points = {}
points[0] = list(np.random.randint(low=0,high=20,size=(10,2)))
points[1] = list(np.random.randint(low=0,high=20,size=(10,2)))
# testing point p(x,y)
# Number of neighbours
k = 3
col = {0:'red',1:'blue'}
for x in points:
    for y in points[x]:
      plt.scatter(y[0],y[1],c = col[x])
plt.title("Existing points")
plt.show()
for x in points:
    print(x)
    for y in points[x]:
      plt.scatter(y[0],y[1],c = col[x])
print("Enter a value for the new point p")
p = [0,0]
p[0] = float(input())
p[1] = float(input())
plt.scatter(p[0],p[1],c = 'green',marker='*',s= 500)
plt.title("New Point")
plt.show()
print("Classifying point is:",classify(points,p,k))
li = classify(points,p,k)
print('Thus the point belongs in the group',li," consisting of the colour : ",col[li])
points[li].append(p)
for x in points:
    print(x)
    for y in points[x]:
      plt.scatter(y[0],y[1],c = col[x])
plt.show()