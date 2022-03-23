import pandas as pd
import numpy as np
import matplotlib.pyplot as mpl
import statistics

# Steffanie Kristiansson
# Jakob Persson

# Importing the data 
happinessData = pd.read_csv("happiness-cantril-ladder.csv")
eduData = pd.read_csv("education.csv")
corruptionData = pd.read_csv("corruption.csv")

# Cleaning up the data
del happinessData["Code"]
del eduData["Code"]
happinessData = happinessData.loc[happinessData["Year"] == 2010]
eduData = eduData.loc[eduData["Year"] == 2010]
corruptionData = corruptionData.loc[corruptionData["Year"] == 2012]

# Merge datasets in case there are unmatched pairs
plottable = pd.merge(happinessData.dropna(), eduData.dropna(), how="outer", on=["Entity", "Entity"])

mpl.scatter(plottable["Barro-Lee: Percentage of population age 15+ with tertiary schooling. Completed Tertiary"], plottable["Life satisfaction in Cantril Ladder (World Happiness Report 2021)"])
mpl.ylabel("Life satisfaction, 0-10 (0 is low, 10 is high)")
mpl.xlabel("Share of the population with a tertiary education")

mpl.show()

plottable = pd.merge(happinessData.dropna(), corruptionData.dropna(), how="outer", on=["Entity", "Entity"])

mpl.scatter(plottable["Corruption Perception Index - Transparency International (2018)"], plottable["Life satisfaction in Cantril Ladder (World Happiness Report 2021)"])
mpl.ylabel("Life satisfaction, 0-10 (0 is low, 10 is high)")
mpl.xlabel("Corruption, 0-100 (0 is high, 100 is low)")

mpl.show()

plottable = pd.merge(corruptionData.dropna(), eduData.dropna(), how ="outer", on=["Entity", "Entity"])

mpl.scatter(plottable["Barro-Lee: Percentage of population age 15+ with tertiary schooling. Completed Tertiary"], plottable["Corruption Perception Index - Transparency International (2018)"])
mpl.ylabel("Corruption, 0-100 (0 is high, 100 is low)")
mpl.xlabel("Share of the population with a tertiary education")

mpl.show()