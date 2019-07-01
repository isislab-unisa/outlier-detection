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
def DBFittizzio():
    Cars = {'Brand': ['Honda Civic','Toyota Corolla','Ford Focus','Audi A4','a','b','c','d','e','f','g','h','i','l'],
            'Price': [22,2500,27,35,34,23,78,65,-4600,78,19,28,12,18]
            }
    df = DataFrame(Cars,columns= ['Brand', 'Price'])
    instances = df.iloc[:, 1].values.reshape(-1, 1)
    from sklearn.neighbors import LocalOutlierFactor
    clf = LocalOutlierFactor(n_neighbors=10, contamination=0.1)
    df['ValoriLOF'] = clf.fit_predict(instances)
    return clf.fit_predict(instances)





'''dfVariazioni = pd.read_csv('2018-Variazioni-Delibere.csv', error_bad_lines=False, sep=';')'''

'''for i,row in  df.iterrows():
    row['Concept']=str(row['Concept'])

for i,row in  df.iterrows():
    row['Concept']=row['Concept'].replace("-","")

df['ValoriNumerici']=df['Concept'].astype(np.int64)'''



def dbMappe():
    dfMappe = pd.read_csv('catastopassicarraichiaiasanferdinandoposillipo.csv', error_bad_lines=False, sep=';',encoding="ISO-8859-1")
    instances= dfMappe.iloc[:,4].values.reshape(-1,1)
    from sklearn.neighbors import LocalOutlierFactor
    clf = LocalOutlierFactor(n_neighbors=22, contamination=0.001)
    y=clf.fit_predict(instances)
    dfMappe['ValoriLOF'] = y
    '''with open('your_file.txt', 'w') as f:
        for item in y.tolist():
            f.write("%s," % item)'''
    return y


def dbDate():
    dfDate = pd.read_csv('sparqlBirthDate', error_bad_lines=False)
    for i, row in dfDate.iterrows():
        row['Concept'] = str(row['Concept'])
    for i, row in dfDate.iterrows():
        row['Concept']=row['Concept'].replace("-", "")

    dfDate['ValoriNumerici']=dfDate['Concept'].astype(np.int64)
    instances= dfDate.iloc[:,0].values.reshape(-1,1)
    from sklearn.neighbors import LocalOutlierFactor
    clf = LocalOutlierFactor(n_neighbors=200, contamination=0.01)
    y=clf.fit_predict(instances)
    dfDate['ValoriLOF'] = y
    '''with open('your_file.txt', 'w') as f:
           for item in y.tolist():
               f.write("%s," % item)'''
    return y

