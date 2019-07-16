# -*- coding: utf-8 -*-
"""Example of using kNN for outlier detection
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function
import os
import sys
import pandas as pd
import numpy as np


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from pandas import DataFrame
def DBFittizzio(neighbors,df):
   # Cars = {'Brand': ['Honda Civic','Toyota Corolla','Ford Focus','Audi A4','a','b','c','d','e','f','g','h','i','l'],
           # 'Price': [22,2500,27,35,34,23,78,65,-4600,78,19,28,12,18]
           # }
   # df = DataFrame(Cars,columns= ['Brand', 'Price'])
    instances = df.iloc[:, 1].values.reshape(-1, 1)
    from sklearn.neighbors import LocalOutlierFactor
    my_list = list()
    clf = LocalOutlierFactor(n_neighbors=neighbors, metric="euclidean")
    df['ValoriLOF'] = clf.fit_predict(instances)
    for i, row in df.iterrows():
        if row['ValoriLOF'] == -1:
            my_list.append(i)
    return my_list





'''dfVariazioni = pd.read_csv('2018-Variazioni-Delibere.csv', error_bad_lines=False, sep=';')'''

'''for i,row in  df.iterrows():
    row['Concept']=str(row['Concept'])

for i,row in  df.iterrows():
    row['Concept']=row['Concept'].replace("-","")

df['ValoriNumerici']=df['Concept'].astype(np.int64)'''



def dbMappe(contamination):
    dfMappe = pd.read_csv('catastopassicarraichiaiasanferdinandoposillipo.csv', error_bad_lines=False, sep=';',encoding="ISO-8859-1")
    instances= dfMappe.iloc[:,4].values.reshape(-1,1)
    from sklearn.neighbors import LocalOutlierFactor
    clf = LocalOutlierFactor(metric='euclidean',contamination=contamination)
    y=clf.fit_predict(instances)
    my_list=list()
    dfMappe['ValoriLOF'] = y
    for i, row in dfMappe.iterrows():
        if row['ValoriLOF'] == -1:
            my_list.append(i)
    return my_list


def dbDate(contamination):
    dfDate = pd.read_csv('sparqlBirthDate', error_bad_lines=False)
    for i, row in dfDate.iterrows():
        row['Concept'] = str(row['Concept'])
    for i, row in dfDate.iterrows():
        row['Concept']=row['Concept'].replace("-", "")

    dfDate['ValoriNumerici']=dfDate['Concept'].astype(np.int64)
    instances= dfDate.iloc[:,0].values.reshape(-1,1)
    from sklearn.neighbors import LocalOutlierFactor
    clf = LocalOutlierFactor(metric="euclidean", contamination=contamination)
    y=clf.fit_predict(instances)
    dfDate['ValoriLOF'] = y
    for i,row in dfDate.iterrows():
        if row['ValoriLOF']==-1:
            print(i, row['Concept'])
    return dfDate


def dbDatasetBellezza(contamination):
    dfBellezza = pd.read_csv('datasetBellezza.csv', error_bad_lines=False,sep=';')


    instances = dfBellezza.iloc[:, 5].values.reshape(-1, 1)
    from sklearn.neighbors import LocalOutlierFactor
    clf = LocalOutlierFactor(metric="euclidean", contamination=contamination)
    y = clf.fit_predict(instances)
    dfBellezza['ValoriLOF'] = y
    for i, row in dfBellezza.iterrows():
        if row['ValoriLOF'] == -1:
            print(i, row['Importo'])
    return dfBellezza


def dbDatasetVariazioniDelibere(contamination):
    dfVariazioni = pd.read_csv('2018-Variazioni-Delibere.csv', error_bad_lines=False, sep=';')

    instances = dfVariazioni.iloc[:, 6].values.reshape(-1, 1)
    from sklearn.neighbors import LocalOutlierFactor
    clf = LocalOutlierFactor(metric="euclidean", contamination=contamination)
    y = clf.fit_predict(instances)
    dfVariazioni['ValoriLOF'] = y
    myList=list()
    for i, row in dfVariazioni.iterrows():
        if row['ValoriLOF'] == -1:
            print(i, row["Unita'OrganizzativaDirigenziale"])
            myList.append(i)
    return myList