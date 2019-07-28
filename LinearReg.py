import numpy as np
from statistics import mean

print("Enter two data sets of same size ")
print("Specify size ")
m = int(input())
x1_data = [int(x1) for x1 in input().split()]
y_data = [int(y) for y in input().split()]
if len(x1_data) != len(y_data) :
    exit("Wrong Input")
x1_dataset = np.array(x1_data)
y_dataset = np.array(y_data)
print(x1_dataset)
print(y_dataset)
x1_dataset = x1_dataset/(max(x1_data) - min(x1_data))
y_dataset = y_dataset/(max(y_data) - min(y_data))


def hyp(x1, theta0, theta1):
    hypo = theta0 + theta1*x1
    return hypo


def cost_func(arr1, arr2, m):
    cost = sum((arr1 - arr2)) / (2 * m)
    return cost


arr = list()
th0 = -0.2
th1 = -0.2
th2 = -0.2

for i in range(15000):
    for a in range(m):
        arr = arr + [hyp(x1_dataset[a], th0, th1)]
    list1 = np.array(arr)
    temp1 = th0 - 0.01 * 2 * cost_func(list1, y_dataset, m)
    temp2 = th1 - 0.01 * 2 * cost_func(list1, y_dataset, m) * sum(x1_dataset)
    th0 = temp1
    th1 = temp2
    del arr[:]
print("Enter two value:\n")
num1 = int(input())
output=hyp(num1/(max(x1_data) - min(x1_data)),th0,th1)*(max(y_data) - min(y_data))
print("The output for it is :\n", output)

