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
import seaborn as sns
from datetime import datetime







def outliers_modified_z_score(ys):
    threshold = 3.5

    median_y = np.median(ys)
    median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys])
    modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y
                         for y in ys]
    df=caricaDB()
    df["Modified-Z-Score"]=modified_z_scores

    return np.where(np.abs(modified_z_scores) > threshold)


def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))

def outliers_z_score(ys):
    threshold = 3

    mean_y = np.mean(ys)
    stdev_y = np.std(ys)
    z_scores = [(y - mean_y) / stdev_y for y in ys]
    df=caricaDB()
    df["Z-Score"]=z_scores
    return np.where(np.abs(z_scores) > threshold)

def stampa():
    df = pd.read_csv('query.csv',error_bad_lines=False,sep=",")

    print(df)
for i,row in  df.iterrows():
    row['d']=str(row['d'])
for i,row in  df.iterrows():
    row['d']=row['d'].replace("-","")
for i,row in  df.iterrows():
    row['d']=row['d'].replace(":","")
for i,row in  df.iterrows():
    row['d']=row['d'].replace("T000000Z","")

    df['ValoriNumerici']=df['d'].astype(np.int64)
    outlier_datapoints = outliers_z_score(df["ValoriNumerici"])
    outlier_datapoints2 = outliers_modified_z_score(df["ValoriNumerici"])
    outlier_datapoints3 = outliers_iqr(df["ValoriNumerici"])
    print("Per Z-Score",outlier_datapoints)
    print("Per Modified Z-Score ",outlier_datapoints2)
    print("Per IQR ",outlier_datapoints3)


stampa()
