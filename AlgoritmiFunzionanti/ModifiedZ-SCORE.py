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
dataset = pd.read_csv('sparqlBirthDate.csv',error_bad_lines=False,sep=";")
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



def outliers_modified_z_score(ys):
    threshold = 3.5

    median_y = np.median(ys)
    median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys])
    modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y
                         for y in ys]
    print(modified_z_scores)
    return np.where(np.abs(modified_z_scores) > threshold)


outlier_datapoints = outliers_modified_z_score(dataset["Longitudine"])
print(outlier_datapoints)


