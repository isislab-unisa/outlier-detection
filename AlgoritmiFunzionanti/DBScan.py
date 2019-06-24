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

df = pd.read_csv('catastopassicarraichiaiasanferdinandoposillipo.csv',error_bad_lines=False,sep=";",encoding="ISO-8859-1")
'''for i,row in  df.iterrows():
        row['Concept']=str(row['Concept'])

for i,row in  df.iterrows():
        row['Concept']=row['Concept'].replace("-","")

df['ValoriNumerici']=df['Concept'].astype(np.int64)
from pandas import DataFrame

Cars = {'Brand': ['m','n','o','p','a','b','c','d','e','f','g','h','i','l'],
        'Price': [22,25000000000000000000000000,27,35,34,-23000000000000000000000000,78,65,46,78,19,28,12,18]
        }

df = DataFrame(Cars,columns= ['Brand', 'Price'])'''
print(df)
X= df.iloc[:,4].values.reshape(-1,1)
'''y_truth=dataset.iloc[:,4].values.reshape(-1,1)'''

norm_data=MinMaxScaler()
X=norm_data.fit_transform(X)


norm_data=MinMaxScaler()
X=norm_data.fit_transform(X)

dbscan=DBSCAN(eps=0.00767337396,min_samples=5)

y=dbscan.fit_predict(X)
print(y)
cluster_labels=np.unique(y)
df['labels']=y
cluster=[]
for index in range (0,len(cluster_labels)):
    lista=[]
    cluster.append(lista)

for index in range (0,len(X)):
    cluster[y[index]].append(X[index])

print("labels:",cluster_labels)
print("elementi",len(cluster[-1]))



