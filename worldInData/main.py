import pandas as pd
import numpy as np
import matplotlib.pyplot as mpl
import statistics

# Steffanie Kristiansson
# Jakob Persson

# Import data 
gdpData = pd.read_csv('gdp-per-capita-worldbank.csv')
lifeData = pd.read_csv('life-expectancy.csv')

# Remove the data we don't want from either dataset. 
# The data from the year 2019 is chosen since it's the most recent year that is included in both sets. 
del gdpData["Code"]
gdpData = gdpData.loc[gdpData["Year"] == 2019]

del lifeData["Code"]
lifeData = lifeData.loc[lifeData["Year"] == 2019]

# Merge the datasets in case there unmatched pairs in the datasets. Before the merge the missing values are discarded since they are of no use. 
plottable = pd.merge(gdpData.dropna(), lifeData.dropna(), how="outer", on=["Entity", "Entity"])

# Plot the data using matplotlib
mpl.scatter(plottable["GDP per capita, PPP (constant 2017 international $)"], plottable["Life expectancy"])
mpl.ylabel("Life Expectancy")
mpl.xlabel("GDP Per Capita")

mpl.show()

# Finding the mean life expectancy as well as the standard deviation. 
deviation = statistics.stdev(plottable.dropna()["Life expectancy"])
mean = statistics.mean(plottable.dropna()["Life expectancy"])

print(deviation)
print(mean)

# Selecting the entries for which the life expectancy is larger than the mean + standard deviation. 
temp = plottable.dropna().loc[plottable["Life expectancy"] > (deviation + mean)]

print("List of countries with a life expectancy at least one standard deviation above the mean")
print(temp)

# get 'low' GDP (under mean ???)
gdpMean = statistics.mean(plottable.dropna()["GDP per capita, PPP (constant 2017 international $)"])

print(gdpMean)

# high life but low capita
lowCapita = plottable.dropna().loc[plottable["GDP per capita, PPP (constant 2017 international $)"] < gdpMean]
lowCapita = pd.merge(lowCapita, temp, how="inner", on=["Entity", "Entity"])

print(lowCapita)