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

'''Cars = {'Brand': ['Honda Civic','Toyota Corolla','Ford Focus','Audi A4','a','b','c','d','e','f','g','h','i','l'],
        'Price': [22,25000000000000000000000000,27,35,34,23,88,35,36,38,19,28,12,18]
        }
Cars = {'Brand': ['Honda Civic','Toyota Corolla','Ford Focus','Audi A4','a','b','c','d','e','f','g','h','i','l'],
        'Price': [22,25000000000000000000000000,27,35,34,23,78,65,46000000000000000000,78,19,28,12,18]
        }
df = DataFrame(Cars,columns= ['Brand', 'Price'])'''

df = pd.read_csv('2018-Variazioni-Delibere.csv',error_bad_lines=False,sep=';',nrows=200)

'''for i,row in  df.iterrows():
    row['Concept']=str(row['Concept'])

for i,row in  df.iterrows():
    row['Concept']=row['Concept'].replace("-","")

df['ValoriNumerici']=df['Concept'].astype(np.int64)'''




instances= df.iloc[:,6].values.reshape(-1,1)
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors=2, contamination=0.2)
# use fit_predict to compute the predicted labels of the training samples
# (when LOF is used for outlier detection, the estimator has no predict,
# decision_function and score_samples methods).
df['ValoriLOF'] = clf.fit_predict(instances)





