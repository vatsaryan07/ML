import pandas as pd
import numpy as np
from sys import argv                                                          #For Command Line Run while passing 'Weather.csv' in the terminal
script, arg1 = argv
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x = pd.read_csv("Weather.csv")
x_dataset = x["MinTemp"].values.reshape(-1,1)
y_dataset = x["MaxTemp"].values.reshape(-1,1)
x_train, x_test, y_train, y_test = train_test_split(x_dataset,y_dataset,test_size=0.3)
reg = LinearRegression()
reg.fit(x_train,y_train)
y_pr = reg.predict(x_test)
lol = pd.DataFrame({"Actual":y_test.flatten(),"Predicted":y_pr.flatten()},index=np.arange(len(y_test)))
print(lol)
mae = metrics.mean_absolute_error(y_test,y_pr)
mse = metrics.mean_squared_error(y_test,y_pr)
rms = np.sqrt(mse)
print("Mean Absolute Error:",mae,"\nMean Squared Error:",mse,"\nRoot Mean Square Error:",rms)