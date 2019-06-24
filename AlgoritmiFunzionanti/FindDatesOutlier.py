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
def caricaDB():
    df = pd.read_csv('sparqlBirthDate',error_bad_lines=False,sep=",")
    '''df=df.iloc[:,4].values.reshape(-1,1)'''
    '''df['Concept']=df['Concept'].astype('datetime64[ns]')'''
    '''df['Concept']=pd.to_datetime(df['Concept'], infer_datetime_format=True,errors='coerce')'''
    print(df)
    for i,row in  df.iterrows():
        row['Concept']=str(row['Concept'])

    '''for i,row in  df.iterrows():
    row['Concept']=datetime.strptime(row['Concept'],'%Y-%m-%d')
    print(df.Concept)

    for i,row in  df.iterrows():
    row['Concept']= int(round(row['Concept'].timestamp() * 1000))
    print(df.Concept)'''
    for i,row in  df.iterrows():
        row['Concept']=row['Concept'].replace("-","")

    df['ValoriNumerici']=df['Concept'].astype(np.int64)

    sns.boxplot(x=df['ValoriNumerici'])

    '''for index,row in df.iterrows():
    row[""data_ora""]= int(round(row[""data_ora""].timestamp() * 1000))
    print(row[""data_ora""])'''

    print(df.ValoriNumerici)

    '''sns.boxplot(x=""descrizione"", y=""data_ora"", data=df)
    np.random.seed(1111)

    grp = df.groupby(['stazione',df.data_ora.dt.dayofweek])['valore']
    df['zscore'] = grp.transform( lambda x: (x-x.mean())/x.std())
    df[ df['zscore'].abs() > 1.5 ]'''
    return df


def outliers_modified_z_score(ys):
    threshold = 3.5

    median_y = np.median(ys)
    median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys])
    modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y
                         for y in ys]
    df=caricaDB2()
    df["Modified-Z-Score"]=modified_z_scores

    return np.where(np.abs(modified_z_scores) > threshold)

def caricaDB2():
    df = pd.read_csv('queryBirthDateWikidata.csv',error_bad_lines=False,sep=",")
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
    sns.boxplot(x=df['ValoriNumerici'])
    return df

def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))

def outliers_z_score(ys,df):
    threshold = 3
    mean_y = np.mean(ys)
    stdev_y = np.std(ys)
    z_scores = [(y - mean_y) / stdev_y for y in ys]
    df["Z-Score"]=z_scores
    return np.where(np.abs(z_scores) > threshold)

def stampa():
    df=caricaDB()
    outlier_datapoints = outliers_iqr(df["ValoriNumerici"])
    print("Per Z-Score",outlier_datapoints)
    return df

def checkOutlier():
    df=caricaDB2()
    outlier_datapoints =outliers_iqr(df["ValoriNumerici"])
    print("Per IQR",outlier_datapoints)
    return df




