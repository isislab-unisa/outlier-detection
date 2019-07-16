import pandas as pd
from AlgoritmiFunzionanti import lofBuono
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import pyplot as plt



def get_dataset():
	data =  pd.read_csv('catastopassicarraichiaiasanferdinandoposillipo.csv',error_bad_lines=False,sep=";",encoding="ISO-8859-1")
	return data


def lof():
    data = pd.read_csv('catastopassicarraichiaiasanferdinandoposillipo.csv', error_bad_lines=False, sep=";",
                       encoding="ISO-8859-1")
    X = data.iloc[:,1:4 ]
    normal,outliers=lofBuono.local_outlier_factor(20,1.5)
    return normal,outliers
