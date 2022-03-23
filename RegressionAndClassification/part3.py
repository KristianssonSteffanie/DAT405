# ASSIGNMENT 2 part 3 
# Jakob Persson, Steffanie Kristiansson

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import seaborn as sns

# Load the dataset
iris = datasets.load_iris()

# A function for KNN-classifying to make testing different k-values and weights easier
def neighbors(k, weight):

  # Splits the dataset into testing and training sets
  xTrain, xTest, yTrain, yTest = train_test_split(iris.data, iris.target, test_size = 0.3, random_state = 0)
  # Creates a classifier depending on the k-value and weight provided in the function call
  knn = KNeighborsClassifier(n_neighbors=k, weights=weight) 

  # Training the model
  knn.fit(xTrain, yTrain)

  # Making predictions on the test set
  yPred = knn.predict(xTest)

  # Tests the accuracy of the model and prints it for human evaluation
  accuracy = metrics.accuracy_score(yTest, yPred)
  print(weight, k)
  print("Accuracy: ", accuracy)
  print(" ")

  # Creating a confusion matrix to visualize how the model performed on the test. 
  cm = metrics.confusion_matrix(yTest, yPred)
  sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap="Blues_r");
  plt.ylabel("Actual label")
  plt.xlabel("Predicted label")
  all_sample_title = "Accuracy score: {0}".format(accuracy)
  plt.title(all_sample_title, size = 15)
  plt.show()

# Function calls that tests different k-values and weights
neighbors(1, "uniform")
neighbors(5, "uniform")
neighbors(20, "uniform")
neighbors(30, "uniform")
neighbors(50, "uniform")
neighbors(75, "uniform")

neighbors(1, "distance")
neighbors(5, "distance")
neighbors(20, "distance")
neighbors(30, "distance")
neighbors(50, "distance")
neighbors(75, "distance")