import numpy as np
from statistics import mean
print("Enter three data sets of same size"

      "Two input data sets"

      "One output data sets")
print("Specify size ")
m = int(input())
x1_data = [int(x1) for x1 in input().split()]
x2_data = [int(x2) for x2 in input().split()]
y_data = [int(y) for y in input().split()]
if len(x1_data) != len(y_data) or len(x2_data) != len(y_data):
    exit("Wrong Input")
x1_dataset = np.array(x1_data)
x2_dataset = np.array(x2_data)
y_dataset = np.array(y_data)
print(x1_dataset)
print(x2_dataset)
print(y_dataset)
x1_dataset = x1_dataset/(max(x1_data) - min(x1_data))
x2_dataset = x2_dataset/(max(x2_data) - min(x2_data))
y_dataset = y_dataset/(max(y_data) - min(y_data))


def hyp(x1, x2, theta0, theta1, theta2):
    hypo = theta0 + theta1*x1 + theta2*x2
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
        arr = arr + [hyp(x1_dataset[a], x2_dataset[a], th0, th1, th2)]
    list1 = np.array(arr)
    temp1 = th0 - 0.001 * 2 * cost_func(list1, y_dataset, m)
    temp2 = th1 - 0.001 * 2 * cost_func(list1, y_dataset, m) * sum(x1_dataset)
    temp3 = th2 - 0.001 * 2 * cost_func(list1, y_dataset, m) * sum(x2_dataset)
    th0 = temp1
    th1 = temp2
    th2 = temp3
    del arr[:]
print("Enter two value:\n")
num1 = int(input())
num2 = int(input())
print("The output for it is :\n", hyp(num1/(max(x1_data) - min(x1_data)),num2/(max(x2_data) - min(x2_data)),th0,th1,th2)*(max(y_data) - min(y_data)))

