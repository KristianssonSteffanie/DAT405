import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stat
from sklearn.cluster import KMeans

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


# perform K-means clustering
# specify the number of clusters and fit the data X
kmeans = KMeans(n_clusters=4, random_state=0).fit(data.reshape(-1,2))

# get the cluster centroids
print(kmeans.cluster_centers_)
print(kmeans.labels_)

# colors for the plot
LABEL_COLOR_MAP = {0 : 'c',
                   1 : 'limegreen',
                   2 : 'darkorange',
                   3 : 'violet'
                   }

label_color = [LABEL_COLOR_MAP[l] for l in kmeans.labels_]

plt.scatter(data[:, 0], data[:, 1], c=label_color)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x')
plt.title('Data points and cluster centroids')
plt.xlabel("phi")
plt.ylabel("psi")
plt.show()