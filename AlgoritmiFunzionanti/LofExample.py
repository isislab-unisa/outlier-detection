# -*- coding: utf-8 -*-
"""Example of using kNN for outlier detection
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function
import os
import sys

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from AlgoritmiFunzionanti.lof import outliers
from pandas import DataFrame

Cars = {'Brand': ['Honda Civic','Toyota Corolla','Ford Focus','Audi A4','a','b','c','d','e','f','g','h','i','l'],
        'Price': [22,25000000000000000000000000,27,35,34,23,88,35,36,38,19,28,12,18]
        }

df = DataFrame(Cars,columns= ['Brand', 'Price'])

'''df = pd.read_csv('datasetBellezza.csv',error_bad_lines=False,sep=";")

for i,row in  df.iterrows():
    row['Importo']=str(row['Importo'])

for i,row in  df.iterrows():
    row['Importo']=row['Importo'].replace("-","")

df['ValoriNumerici']=df['Importo'].astype(np.int64)'''

instances= df.iloc[:,1].values.reshape(-1,1)
lof = outliers(10, instances)

for outlier in lof:
    print(outlier["lof"],outlier["instance"])

