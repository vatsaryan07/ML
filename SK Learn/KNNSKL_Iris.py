import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn
sn.set()
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
iris_data,iris_class = load_iris(return_X_y=True)
print(iris_data)                                                              # Iris data consists of four columns of sepal length, sepal width, petal length, petal width

iris_xtr, iris_xte,iris_ytr,iris_yte = train_test_split(iris_data,iris_class,test_size=.2)
kclass = KNeighborsClassifier(n_neighbors=8)
kclass.fit(iris_xtr,iris_ytr)
iris_ypred = kclass.predict(iris_xte)
iris_ypred = iris_ypred.flatten()
iris_yte = iris_yte.flatten()
mse = mean_squared_error(iris_yte,iris_ypred)
print("Accuracy : ",mse)
df = pd.DataFrame({"Actual":iris_yte,"Predicted":iris_ypred})
print(df)