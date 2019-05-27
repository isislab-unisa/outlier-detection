import seaborn as sns
import pandas as pd
import numpy as np
titanic = sns.load_dataset('titanic')
titanic = titanic.copy()
titanic = titanic.dropna()
titanic['age'].plot.hist(
  bins = 50,
  title = "Histogram of the age variable"
)
from scipy.stats import zscore
dataset = pd.read_csv('datasetBellezza.csv',error_bad_lines=False,sep=";")
dataset=dataset.copy()
dataset=dataset.dropna()


from pandas import DataFrame

Cars = {'Brand': ['Honda Civic','Toyota Corolla','Ford Focus','Audi A4','a','b','c','d','e','f','g','h','i','l'],
        'Price': [22,25000000000000000000000000,27,35,34,23,78,65,46,78,19,28,12,18]
        }

df = DataFrame(Cars,columns= ['Brand', 'Price'])

print (df)

Cars = {'Brand': ['Honda Civic','Toyota Corolla','Ford Focus','Audi A4'],
        'Price': [22,25000000000000000000000000,27,35]
        }

df = DataFrame(Cars,columns= ['Brand', 'Price'])

print (df)

def outliers_z_score(ys):
    threshold = 3

    mean_y = np.mean(ys)
    stdev_y = np.std(ys)
    z_scores = [(y - mean_y) / stdev_y for y in ys]
    return np.where(np.abs(z_scores) > threshold)

outlier_datapoints = outliers_z_score(df["Price"])
print(outlier_datapoints)

outlier_datapoints = outliers_z_score(dataset["Importo"])
print(outlier_datapoints)

'''dataset[""zscore""] = outliers_z_score(dataset[""Unita'OrganizzativaDirigenziale""])
dataset[""is_outlier""] = dataset[""zscore""]
dataset[dataset[""is_outlier""]]

for index,row in dataset.iterrows():
    if(row[""is_outlier""]==True):  
        print(row[""Unita'OrganizzativaDirigenziale""])'''
