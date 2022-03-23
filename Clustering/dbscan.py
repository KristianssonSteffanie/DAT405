from matplotlib import collections
import collections as coll
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Steffanie Kristiansson
# Jakob Persson

# Sources with data
data200 = pd.read_csv('data_200.csv')
data500 = pd.read_csv('data_500.csv')
allData = pd.read_csv('data_all.csv')

# reshape data to 2d array
psi = np.array(allData["psi"])
phi = np.array(allData["phi"])
data = np.stack((phi, psi), axis=1)
allNames = np.array(allData["residue name"])

# Compute DBSCAN
db = DBSCAN(eps=15, min_samples=35).fit(data)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

# Plot result
# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = data[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = data[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.xlabel("phi")
plt.ylabel("psi")
plt.show()

# create the bar chart
names = []
i = 0
# find the names of all outliers and add them to an array
for point in data:
    if labels[i] == -1:
        names.append(allNames[i])
    i += 1
names = np.array(names)

# count the frequency for each name 
c = coll.Counter(names)
c = sorted(c.items())

# add every item to the plot 
for item in c:
    plt.bar(item[0], item[1])

plt.show()

# assign a color to each label
LABEL_COLOR_MAP = {0 : 'c',
             1 : 'limegreen',
             2 : 'darkorange',
             3 : 'violet',
             4 : 'tomato',
             5 : 'olivedrab',
             6 : 'slategrey',
             7 : 'skyblue',
             -1: 'k'}
label_color = [LABEL_COLOR_MAP[l] for l in labels]
plt.scatter(data[:, 0], data[:, 1], c=label_color)
plt.title('Estimated number of outliers: %d' % n_noise_)
plt.xlabel("phi")
plt.ylabel("psi")
plt.show()
