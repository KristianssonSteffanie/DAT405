import os
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.preprocessing import Binarizer, binarize
from nltk.corpus import stopwords


# Steffanie Kristiansson
# Jakob Persson

# store all the files from choosen directory in a list
def getSubDir(directory):
  contents = []
  for file in os.listdir(directory):
    filename = os.path.join(directory, file)
    if os.path.isfile(filename):
      f = open(filename, encoding="latin-1")
      contents.append(f.read())

  return contents

# function to get dataframe
def getDataFrame(content, label):
  result = []
  for item in content:
    result.append([item.lower(), label])

  return pd.DataFrame(result, columns=["mail", "id"])

# function to get train and test for each directory/category (easy/hard/spam)
def getTrainTest(directory, label):
  content = getSubDir(directory)
  dataframe = getDataFrame(content, label)
  
  train, test = train_test_split(dataframe, test_size=0.3, random_state = 0)

  return train, test

# train and test dataframes
spamtrain, spamtest = getTrainTest("C:/Users/steff/Documents/year2/AI_DAT405/A4/spam", 1)
easytrain, easytest = getTrainTest("C:/Users/steff/Documents/year2/AI_DAT405/A4/easy_ham", 0)
hardtrain, hardtest = getTrainTest("C:/Users/steff/Documents/year2/AI_DAT405/A4/hard_ham", 0)

# merging easy and hard
alltest = pd.concat([easytest, hardtest])
alltrain = pd.concat([easytrain, hardtrain])

# x and y test/train
xTrain = pd.concat([hardtrain['mail'],spamtrain['mail']])
xTest = pd.concat([hardtest['mail'],spamtest['mail']])
yTrain = pd.concat([hardtrain['id'],spamtrain['id']])
yTest = pd.concat([hardtest['id'],spamtest['id']])

# vectorizing
vectorizer = CountVectorizer() # for q4: CountVectorizer(max_df=0.85, min_df=2, stop_words='english')
vectorizer.fit(xTrain)
xTrainV = vectorizer.transform(xTrain)
xTestV = vectorizer.transform(xTest)


mnb = MultinomialNB()
mnb.fit(xTrainV, yTrain)
mnbPred = mnb.predict(xTestV)
confMatrix = confusion_matrix(yTest, mnbPred)
trueneg, falsepos, falseneg, truepos = confMatrix.ravel()
# true negative, false positive, false negative, true positive
print("Accuracy (mnb): ", metrics.accuracy_score(yTest, mnbPred), "\n",
      "TN:",trueneg, "\n", 
      "FP:", falsepos, "\n",
      "FN:",falseneg, "\n", 
      "TP:" ,truepos, "\n",
      "TP rate:", truepos/(truepos+falseneg), "\n",
      "FN rate:",falseneg/(truepos+falseneg))

bnb = BernoulliNB()
bnb.fit(xTrainV, yTrain)
bnbPred = bnb.predict(xTestV)
confMatrix = confusion_matrix(yTest, bnbPred)
trueneg, falsepos, falseneg, truepos = confMatrix.ravel()
# true negative, false positive, false negative, true positive
print("Accuracy (bnb):", metrics.accuracy_score(yTest, bnbPred), "\n",
      "TN:",trueneg, "\n", 
      "FP:", falsepos, "\n",
      "FN:",falseneg, "\n", 
      "TP:" ,truepos, "\n",
      "TP rate:", truepos/(truepos+falseneg), "\n",
      "FN rate:",falseneg/(truepos+falseneg))

# loop through different binarize values on Bernoulli
'''n = range(11)
for i in n:
    bnb = BernoulliNB(binarize=i)
    bnb.fit(xTrainV, yTrain)
    bnbPred = bnb.predict(xTestV)
    print(("Accuracy (bnb):", metrics.accuracy_score(yTest, bnbPred), "\n",
      "TN:",trueneg, "\n", 
      "FP:", falsepos, "\n",
      "FN:",falseneg, "\n", 
      "TP:" ,truepos, "\n",
      "TP rate:", truepos/(truepos+falseneg), "\n",
      "FN rate:",falseneg/(truepos+falseneg))
'''
# which wirds are eliminated, uncommet below
#print(vectorizer.stop_words_)

# to see what's remaning
#print(vectorizer.vocabulary_)