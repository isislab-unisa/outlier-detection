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
import os

cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in '%s': %s" % (cwd, files))
df = pd.read_csv('2018-Variazioni-Delibere.csv',error_bad_lines=False,sep=";")

'''for i,row in  df.iterrows():
        row['Concept']=str(row['Concept'])

for i,row in  df.iterrows():
        row['Concept']=row['Concept'].replace("-","")

df['ValoriNumerici']=df['Concept'].astype(np.int64)
from pandas import DataFrame

Cars = {'Brand': ['Honda Civic','Toyota Corolla','Ford Focus','Audi A4','a','b','c','d','e','f','g','h','i','l'],
        'Price': [22,25000000000000000000000000,27,35,34,23,78,65,46,78,19,28,12,18]
        }

df = DataFrame(Cars,columns= ['Brand', 'Price'])'''
X= df.iloc[:, 6].values.reshape(-1, 1)

'''y_truth=dataset.iloc[:,4].values.reshape(-1,1)'''

norm_data=MinMaxScaler()
X=norm_data.fit_transform(X)
print(X)
nbrs=NearestNeighbors(n_neighbors=2,algorithm='ball_tree').fit(X)

distances, indices=nbrs.kneighbors(X)
distances=np.sort(distances,axis=0)
distances=distances[:,1]
med=np.mean(distances)
print(med)
plot.plot(distances)

plot.show()
