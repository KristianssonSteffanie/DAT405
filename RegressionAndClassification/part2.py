# ASSIGNMENT 2, part 2
# Jakob Persson, Steffanie Kristiansson

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn import metrics

# Load the iris dataset
iris = load_iris()

# Split the dataset into testing and training datasets
xTrain, xTest, yTrain, yTest = train_test_split(iris.data, iris.target, test_size = 0.3, random_state = 0)

# Create a logistic regression model
logisticRegr = LogisticRegression(multi_class="ovr", solver="liblinear")
logisticRegr.fit(xTrain, yTrain)

# Use the model to make predictions on the test set
predictions = logisticRegr.predict(xTest)
# Evaluate the predictions the model made with the test set
score = logisticRegr.score(xTest, yTest)

# Plot the result in a confusion matrix
cm = metrics.confusion_matrix(yTest, predictions)
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap="Blues_r");
plt.ylabel("Actual label")
plt.xlabel("Predicted label")
all_sample_title = "Accuracy score: {0}".format(score)
plt.title(all_sample_title, size = 15)
plt.show()