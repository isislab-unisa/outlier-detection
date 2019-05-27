from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import MinMaxScaler
import pandas
import matplotlib.pyplot as plot
import matplotlib.cm as cm
import numpy as np


dataset = pandas.read_csv('datasetBellezza.csv',error_bad_lines=False,sep="";"")
print(dataset)

X = dataset.iloc[:,5].values.reshape(-1,1)
norm_data=MinMaxScaler()
X=norm_data.fit_transform(X)

print(""Valore di X"",X)

kmeans=KMeans(n_clusters=4,init='k-means++',random_state=0)
y=kmeans.fit_predict(X)

plot.scatter(X[y==0,0],X[y==0,1], s=25 ,c='red',label='cluster1')

plot.scatter(X[y==1,0],X[y==1,1], s=25 ,c='blue',label='cluster2')

plot.scatter(X[y==2,0],X[y==2,1], s=25 ,c='magenta',label='cluster3')

plot.scatter(X[y==3,0],X[y==3,1], s=25 ,c='cyan',label='cluster4')

plot.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=25,c='yellow',label='centroids')
plot.title=('kmeans clustering')
plot.legend()
plot.show()



