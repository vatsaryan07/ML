import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

X, y = load_digits(return_X_y=True)
x1, x2, y1, y2 = train_test_split(X,y,test_size=0.3)
reg = LogisticRegression(random_state=0,
                          multi_class='multinomial',max_iter=4000)
reg.fit(x1, y1)
score = reg.score(x2,y2)
print(score*100," is the efficiency of the program")
y_pr = reg.predict(x2[0:10])
print(y_pr)
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(x2[0:5], y2[0:5])):
 plt.subplot(1, 5, index + 1)
 plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
 plt.title('Training: %i\n' % label, fontsize = 20)
plt.show()
