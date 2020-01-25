import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
sn.set()
data = pd.read_csv("train.csv")
'''
Grouping the dataset with respect to Ticket and Sex and Cabin to check the 
relation it has with survival of the people on the Ship

'''
#print(data.groupby(data["Ticket"]).median())
#print(data.groupby(data["Survived"]).count())
#print(data.loc[:,("PassengerId","Sex","Survived")].groupby(data["Sex"]).mean())
'''
print(data.loc[:,("PassengerId","Sex","Survived")].groupby(data["Cabin"]).count())
njk = 'A23'
print(type(njk))
print(type(data.loc[0,"Cabin"]))
for x in range(len(data)):
    y = data.loc[x,"Cabin"]
    if str(y[0]) == 'A':
        print('Yes')
    elif str(y[0]) == 'B':
        print('No')
    else:
        print(y[0])
'''
'''
We merge the Sibling and Spouse field with the Parents and Children field to 
a Family group to eliminate the need of two different fields. Since the point
of the two fields is to check corelation of the person wrt to any family they 
have, they are merged into one field 
'''
x_test = pd.read_csv('test.csv')
xdt = pd.read_csv('test.csv')
for x in range(len(x_test)):
    y = x_test.loc[x,"SibSp"]
    z = x_test.loc[x,"Parch"]
    x_test.loc[x,"Family"] = int(y)+int(z)

x_test.loc[x_test["Family"] > 0] = 1
#print(x_test.columns)
#print(x_test.iloc[:,1:3])
'''
Embarked field has three inputs: C for Cherbourg, Q for Queenstown and 
S for Southampton. We simply convert the three into numbers for easier
evaluation
'''
for x in range(len(x_test)):
    y = x_test.loc[x,"Embarked"]
    if y == 'S':
        x_test.loc[x,"Embarked"] = 0
    elif y == 'Q':
        x_test.loc[x,"Embarked"] = 1
    else:
        x_test.loc[x,"Embarked"] = 2

'''
Setting bands for the Fare paid by the people boarding the ship, 
since that might have created a disparity in the seating
'''
for x in range(len(x_test)):
    y = x_test.loc[x,"Fare"]
    if y == 0:
        x_test.loc[x,"Fare"] = 0
    elif y < 50:
        x_test.loc[x,"Fare"] = 1
    elif y < 100:
        x_test.loc[x,"Fare"] = 2
    else:
        x_test.loc[x,"Fare"] = 3
#print(x_test["Pclass"])
'''
Setting age bands according to infants, Teenagers, Adults, 
and Old People
'''
for x in range(len(x_test)):
    x1 = x_test.loc[x,"Age"]
    if x1<10:
        x_test.loc[x,"Age"] = 0
    elif x1<20:
        x_test.loc[x,"Age"] = 1
    elif x1<50:
        x_test.loc[x,"Age"] = 2
    elif x1<70:
        x_test.loc[x,"Age"] = 3
    else:
        x_test.loc[x,"Age"] = 4

'''
Setting integer values for male and female
'''
for x in range(len(x_test)):
    x1 = x_test.loc[x,"Sex"]
    if x1 == 'male':
        x_test.loc[x,"Sex"] = 0
    elif x1 == 'female':
        x_test.loc[x,"Sex"] = 1

x_test = x_test.loc[:,("Pclass","Embarked","Sex","Fare","Age","Family")]
#print('Test df',x_test.head())




for x in range(len(data)):
    y = data.loc[x,"SibSp"]
    z = data.loc[x,"Parch"]
    data.loc[x,"Family"] = int(y)+int(z)

data.loc[data["Family"] > 0] = 1
#print(data.columns)
#print(data.iloc[:,1:3])
for x in range(len(data)):
    y = data.loc[x,"Embarked"]
    if y == 'S':
        data.loc[x,"Embarked"] = 0
    elif y == 'Q':
        data.loc[x,"Embarked"] = 1
    else:
        data.loc[x,"Embarked"] = 2

for x in range(len(data)):
    y = data.loc[x,"Fare"]
    if y == 0:
        data.loc[x,"Fare"] = 0
    elif y < 50:
        data.loc[x,"Fare"] = 1
    elif y < 100:
        data.loc[x,"Fare"] = 2
    else:
        data.loc[x,"Fare"] = 3
#print(data["Pclass"])
for x in range(len(data)):
    x1 = data.loc[x,"Age"]
    if x1<10:
        data.loc[x,"Age"] = 0
    elif x1<20:
        data.loc[x,"Age"] = 1
    elif x1<50:
        data.loc[x,"Age"] = 2
    elif x1<70:
        data.loc[x,"Age"] = 3
    else:
        data.loc[x,"Age"] = 4

for x in range(len(data)):
    x1 = data.loc[x,"Sex"]
    if x1 == 'male':
        data.loc[x,"Sex"] = 0
    elif x1 == 'female':
        data.loc[x,"Sex"] = 1
#print(data["Embarked"])
#print(data["Sex"])
#print(data.loc[:,["Survived","Cabin"]])
'''
We do not use PassengerId, Name and Ticket as they are unique and thus counter
productive. We drop the Cabin field as only a few people have documented 
Cabin numbers, and multiple fields being NaN. Thus we drop it too.
'''
survived = data[data["Survived"]==0]
dead = data[data["Survived"] == 1 ]
tk = data.drop("Survived",axis=1)
tk = tk.drop("PassengerId",axis=1)
tk = tk.drop("Name",axis=1)
tk = tk.drop("Ticket",axis=1)
tk = tk.drop("Cabin",axis=1)
#print(tk.columns)
tk = tk.fillna(tk.mean())

plt.subplot(1,2,1)
plt.hist(survived["Age"],bins=30,label="Age")
plt.ylabel("Number of People")
plt.xlabel("Age")
plt.title(label="Survived")
plt.subplot(1,2,2)
plt.hist(dead["Age"],bins=30,label="Age")
plt.title(label="Dead")
plt.xlabel("Age")
plt.show()
plt.subplot(1,2,1)
plt.title(label="Survived")
plt.hist(survived["Pclass"])
plt.ylabel('Number of people')
plt.xlabel('Passenger Class')
plt.subplot(1,2,2)
plt.title(label="Dead")
plt.xlabel('Passenger Class')
plt.hist(dead["Pclass"])
plt.show()
plt.subplot(1,2,1)
plt.title(label="Survived")
plt.hist(survived["Fare"])
plt.ylabel('Number of people')
plt.xlabel('Fare')
plt.subplot(1,2,2)
plt.title(label="Dead")
plt.xlabel('Fare')
plt.hist(dead["Fare"])
plt.show()
#
#print(data.groupby(data["Parch"]).mean())
#print(survived["Embarked"])
#print(dead["Embarked"])
#plt.subplot(1,2,1)
#plt.bar(survived["Parch"])
#plt.hist(survived["Parch"])
#plt.subplot(1,2,2)
#plt.hist(dead["Parch"])
#plt.show()
#'''
bestf = SelectKBest(score_func=chi2,k=7)
fit = bestf.fit(tk,data["Survived"])
scor = pd.DataFrame(fit.scores_)
col = pd.DataFrame(tk.columns)
fs = pd.concat([col,scor],axis=1)
fs.columns = ['Features','Scores']
print(fs)
#'''
'''
Using Passenger class, the modified Embarked, Sex, Fare and age fields 
along with the new Family Field to use make the test and train dataset
'''
test = tk.loc[:,("Pclass","Embarked","Sex","Fare","Age","Family")]
x1,x2,y1,y2 = train_test_split(test,data["Survived"],test_size=0.9)
#print(test.head())
dt = DecisionTreeClassifier()
dt.fit(test,data["Survived"])
y_pr = dt.predict(x_test)
#y_pr = y_pr.reshape(-1,1)
score = round(dt.score(test, data['Survived']) * 100, 2)
print("Confidence Score is : ",score)
y_pred = dt.predict(x2)
scd = accuracy_score(y2,y_pred)
print("Accuracy is : ",scd)
#'''
Y = pd.DataFrame({"PassengerID":xdt["PassengerId"],"Survived":y_pr})
Y.to_csv('Sub.csv',index=False)