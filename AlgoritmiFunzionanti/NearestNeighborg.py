from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plot
import matplotlib.cm as cm
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN

from sklearn.neighbors import NearestNeighbors

df = pd.read_csv('..\DB\sparql',error_bad_lines=False)
for i,row in  df.iterrows():
        row['Concept']=str(row['Concept'])

for i,row in  df.iterrows():
        row['Concept']=row['Concept'].replace("-","")

df['ValoriNumerici']=df['Concept'].astype(np.int64)
X= df.iloc[:,1].values.reshape(-1,1)
'''y_truth=dataset.iloc[:,4].values.reshape(-1,1)'''

norm_data=MinMaxScaler()
X=norm_data.fit_transform(X)

nbrs=NearestNeighbors(n_neighbors=10,algorithm='ball_tree').fit(X)

distances, indices=nbrs.kneighbors(X)
distances=np.sort(distances,axis=0)
distances=distances[:,1]

plot.plot(distances)

plot.show()
