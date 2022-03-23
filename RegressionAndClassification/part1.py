# ASSIGNMENT 2, part 1. 
# Jakob Persson, Steffanie Kristiansson

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from yellowbrick.regressor import ResidualsPlot

# Read the data from the csv file
dataset = pd.read_csv("bopriser.csv")
# Extract the data we want
priceData = dataset["pris"]
areaData = dataset["kvm"]

# Convert the data to numpy arrays and then reshape the area array into 2d 
area = np.array(areaData)
price = np.array(priceData)
area = area.reshape(area.size, 1)

# Plot the data
plt.scatter(area, price)
plt.show()

# Find a slope and intercept with linear regression on the data
model = LinearRegression().fit(area, price)
slope = model.coef_
print(slope)

intercept = model.intercept_
print(intercept)

# Using the model to predict the prices of 100, 150 and 200 m2 houses
hundred = model.predict([[100]])
hundredfifty = model.predict([[150]])
twohundred = model.predict([[200]])
print("100 m2: ", hundred)
print("150 m2: ", hundredfifty)
print("200 m2: ", twohundred)


# Drawing a residual plot, taken from the sci-kit website. 
visualizer = ResidualsPlot(model, hist=False)
visualizer.fit(area, price)  
visualizer.score(area, price)
visualizer.show()