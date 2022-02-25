"""
Name: Luciano Zavala
Date: 02/25/22
Assignment: Module 8: Classification Project
Due Date: 02/27/22
About this project: python script that computes different classification models and analyze them.
Assumptions:NA
All work below was performed by LZZ
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot
from numpy import where
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

df = pd.read_csv("mushrooms.csv")

# x values array creation
X = df.loc[:, df.columns != 'class']

# data wrangling
X = pd.get_dummies(X)
X = pd.DataFrame(X.values.astype('float32'))
X.to_csv("X.csv")
X = np.array(X)

# y values array creation
Y = df['class']

# data wrangling
Y.replace('p', 0, inplace=True)
Y.replace('e', 1, inplace=True)
Y = Y.values.astype('float32')
Y_graph = pd.DataFrame(Y)
Y = np.array(Y)

# split the data into the train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# ***********************************************************************************

# K-nearest neighbor model
KNN = KNeighborsClassifier(n_neighbors=2)
KNN.fit(X_train, Y_train)
pred = KNN.predict(X_test)
print('\nKNN Model accuracy score:', accuracy_score(Y_test, pred))
print("This model seem to get a 100% precision score while trying to "
      "predict if a mushroom is either poisonous or edible. This means "
      "that theres no errors in the predictions so far")

# ***********************************************************************************

# Naive Bayes model
NBayes = GaussianNB().fit(X_train, Y_train)
pred = NBayes.predict(X_test)
print('\nNaive Bayes Model accuracy score:', accuracy_score(Y_test, pred))
print("This model seem to get a 100% precision score while trying to "
      "predict if a mushroom is either poisonous or edible. For this"
      " model the score was a little lower of 0.94 meaning there was some error.")

# ***********************************************************************************

# Logistic regression model
model = LogisticRegression(solver='lbfgs',
                           multi_class='multinomial',
                           max_iter=200).fit(X_train, Y_train)
pred = model.predict(X_test)
print('\nLogistic Regression Model accuracy score:', accuracy_score(Y_test, pred))
print("This model seem to get a 100% precision score while trying to "
      "predict if a mushroom is either poisonous or edible. This means "
      "that theres no errors in the predictions so far")

# ***********************************************************************************

# Neural network model
model = MLPClassifier(max_iter=1000).fit(X_train, Y_train)
pred = model.predict(X_test)
print('\nNeural network Model accuracy score:', accuracy_score(Y_test, pred))
print("This model seem to get a 100% precision score while trying to "
      "predict if a mushroom is either poisonous or edible. This means "
      "that theres no errors in the predictions so far")

# ***********************************************************************************

print("\nModel comparison:")
print("The first K-nearest neighbor model does the job exceptionally with a 1.0 precision score.")
print("The second model using a Naive Bayes model gets a 0.94 precision score. Its the lowest score "
      "and I would not use this model for this task.")
print("The third model is based using a Logistic Regresion algorithm and does the job exceptionally with a 1.0"
      "precision score.")
print("finally, the last model is based using a Neural network and get a 1.0 precision score.")

# create scatter plot for samples from each class
for class_value in range(2):
    # get row indexes for samples with this class
    row_ix = where(Y_graph == class_value)
    # create scatter of these samples
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show()

