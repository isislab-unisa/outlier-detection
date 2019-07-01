from sklearn.datasets import make_blobs


from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import MinMaxScaler
import pandas
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

dataset = pandas.read_csv('datasetBellezza.csv',error_bad_lines=False,sep=";")
print(dataset)
X = dataset.iloc[:,5].values.reshape(1,-1)
print(X)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=1)
kmeans.fit(X)


f, ax = plt.subplots(figsize=(7, 5))
ax.set_title("Blob")
ax.scatter(X[:, 0], X[:, 1], label='Points')
ax.scatter(kmeans.cluster_centers_[:, 0],
               kmeans.cluster_centers_[:, 1], label='Centroid',
               color='r')
ax.legend()
plt.show()
'''distances = kmeans.transform(X)
print(""distanze"",distances)
sorted_idx = np.argsort(distances.ravel())[::-1][:1]
print(""Sorted idx"",sorted_idx)

f, ax = plt.subplots(figsize=(7, 5))
ax.set_title(""Single Cluster"")
ax.scatter(X[:, 0], X[:, 1], label='Points')
ax.scatter(kmeans.cluster_centers_[:, 0],
               kmeans.cluster_centers_[:, 1],
               label='Centroid', color='r')
ax.scatter(X[sorted_idx][:, 0], X[sorted_idx][:, 1],
               label='Extreme Value', edgecolors='g',
               facecolors='none', s=100)
ax.legend(loc='best')
plt.show()'''
