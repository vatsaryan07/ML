import pandas as pd
import numpy as np
from sys import argv
script, arg1 = argv
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x = pd.read_csv("winequality.csv")
attr = x[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates','alcohol']].values
lab = x["quality"].values
x_train, x_test, y_train,y_test = train_test_split(attr,lab,test_size=0.3)
reg = LinearRegression()
reg.fit(x_train,y_train)
y_pr = reg.predict(x_test)
df = pd.DataFrame({"Actual":y_test.flatten(),"Predicted":y_pr.flatten()})
print(df)
