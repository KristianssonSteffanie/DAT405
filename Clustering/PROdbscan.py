import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics

# Jakob Persson
# Steffanie Kristiansson

# Sorces with data
data200 = pd.read_csv("data_200.csv")
data500 = pd.read_csv("data_500.csv")
dataAll = pd.read_csv("data_all.csv")

# the x and y labels on the graph
plt.ylabel("phi")
plt.xlabel("psi")

# reshape data to 2d array
dataPro = dataAll.loc[dataAll["residue name"] == "PRO"]
proPhi = np.array(dataPro["phi"])
proPsi = np.array(dataPro["psi"])

# join arrays along a new axis
data = np.stack((proPhi, proPsi), axis = 1)

# Compute dbscan
db = DBSCAN(eps=19, min_samples=40).fit(data)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# number of clusters in labels, ignoring noise if present
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

# plot result, black removed and used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
  if k == -1:
    col = [0, 0, 0, 1]
  
  class_member_mask = (labels == k)

  xy = data[class_member_mask & core_samples_mask]
  plt.plot(xy[:,0], xy[:,1], 'o', mfc=tuple(col), mec='k',markersize=14)

  xy = data[class_member_mask & ~core_samples_mask]
  plt.plot(xy[:, 0], xy[:, 1], 'o', mfc=tuple(col), mec='k', markersize=6)

plt.title('Estimated number of clusters for PRO: %d' % n_clusters_)
plt.show() 
