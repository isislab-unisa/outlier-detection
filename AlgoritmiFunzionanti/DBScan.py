from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plot
import matplotlib.cm as cm
import numpy as np
from sklearn import datasets
from sklearn.cluster import DBSCAN

def cluster_evaluate(cluster):
    count=np.bincount(cluster)
    count=np.argmax(count)

    purity=np.count_nonzero(cluster==count)/np.size(cluster)
    return purity

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


norm_data=MinMaxScaler()
X=norm_data.fit_transform(X)

dbscan=DBSCAN(eps=0.0007)

y=dbscan.fit_predict(X)

cluster_labels=np.unique(y)

cluster=[]
for index in range (0,len(cluster_labels)):
    lista=[]
    cluster.append(lista)

for index in range (0,len(X)):
    cluster[y[index]].append(X[index])

print("labels:",cluster_labels)
print("elementi",len(cluster[-1]))



