import numpy as np
import matplotlib.pyplot as plt
def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))

import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import zscore
dataset = pd.read_csv('catastopassicarraichiaiasanferdinandoposillipo.csv',error_bad_lines=False,sep=";",encoding="ISO-8859-1")
dataset=dataset.copy()
dataset=dataset.dropna()
from pandas import DataFrame

Cars = {'Brand': ['Honda Civic','Toyota Corolla','Ford Focus','Audi A4','a','b','c','d','e','f','g','h','i','l'],
        'Price': [22,25000000000000000000000000,27,35,34,23,78,65,46000000000000000,78,19,28,12,18]
        }

df = DataFrame(Cars,columns= ['Brand', 'Price'])

print (df)

Cars = {'Brand': ['Honda Civic','Toyota Corolla','Ford Focus','Audi A4'],
        'Price': [22,25000000000000000000000000,27,35]
        }

df2 = DataFrame(Cars,columns= ['Brand', 'Price'])

print (df2)

sns.boxplot(x=dataset["Longitudine"])


outlier_datapoints = outliers_iqr(dataset["Longitudine"])
print(outlier_datapoints)
