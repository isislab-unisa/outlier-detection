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

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from AlgoritmiFunzionanti.lof import outliers

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



'''dfMappe = pd.read_csv('catastopassicarraichiaiasanferdinandoposillipo.csv', error_bad_lines=False, sep=';', encoding="ISO-8859-1")

dfVariazioni = pd.read_csv('2018-Variazioni-Delibere.csv', error_bad_lines=False, sep=';')'''

'''for i,row in  df.iterrows():
    row['Concept']=str(row['Concept'])

for i,row in  df.iterrows():
    row['Concept']=row['Concept'].replace("-","")

df['ValoriNumerici']=df['Concept'].astype(np.int64)'''




'''instances= dfVariazioni.iloc[:,6].values.reshape(-1,1)
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors=100, contamination=0.01)
# use fit_predict to compute the predicted labels of the training samples
# (when LOF is used for outlier detection, the stimator has no predict,
# decision_function and score_samples methods).
dfVariazioni['ValoriLOF'] = clf.fit_predict(instances)

dfDate = pd.read_csv('sparqlBirthDate', error_bad_lines=False)
for i, row in dfDate.iterrows():
    row['Concept'] = str(row['Concept'])
for i, row in dfDate.iterrows():
    row['Concept']=row['Concept'].replace("-", "")

dfDate['ValoriNumerici']=dfDate['Concept'].astype(np.int64)
instances= dfDate.iloc[:,0].values.reshape(-1,1)
clf = LocalOutlierFactor(n_neighbors=200, contamination=0.01)
# use fit_predict to compute the predicted labels of the training samples
# (when LOF is used for outlier detection, the stimator has no predict,
# decision_function and score_samples methods).
dfDate['ValoriLOF'] = clf.fit_predict(instances)'''

