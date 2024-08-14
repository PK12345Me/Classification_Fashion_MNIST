import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

test = pd.read_csv('fashion-mnist_test.csv')
train = pd.read_csv('fashion-mnist_train.csv')

#since train and test data is already split, makes it easier to split it into train and test
x_train = train.drop(columns = 'label')
y_train = train['label']

x_test = test.drop(columns = 'label')
y_test = test['label']

# to display what the images look like

plt.imshow(np.reshape(x_train_list[0],(28, 28)))
plt.show()

# invoking the logistic regression model
logit = LogisticRegression(random_state=42,max_iter = 1000).fit(x_train,y_train)

#computing train and test score - they should NOT be very far apart
logit.score(x_train,y_train)
logit.score(x_test,y_test)

# using .predict() method to predict/classify value of y for a given x-series

logit.predict([x_train_list[49999]])

# it generates output like this

#  UserWarning: X does not have valid feature names, but LogisticRegression was fitted with feature names
#   warnings.warn(
# array([9], dtype=int64)
# number 9 - is for high-ankle boots

plt.imshow(np.reshape(x_train_list[49999],(28, 28)))
plt.show()

# when we generate image using all the response variables in list #49999 it generates the correct result
