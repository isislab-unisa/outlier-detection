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
dataset = pd.read_csv('sparqlBirthDate',error_bad_lines=False,sep=";")
dataset=dataset.copy()
dataset=dataset.dropna()
from pandas import DataFrame

dfVariazioni = pd.read_csv('2018-Variazioni-Delibere.csv', error_bad_lines=False, sep=';')
Cars = {
    'Brand': ['Honda Civic', 'Toyota Corolla', 'Ford Focus', 'Audi A4', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
              'l'],
    'Price': [22, 2500, 27, 35, 34, 23, 78, 65, -4600, 78, 19, 28, 12, 18]
}
df = DataFrame(Cars, columns=['Brand', 'Price'])
print (df)

Cars = {'Brand': ['Honda Civic','Toyota Corolla','Ford Focus','Audi A4'],
        'Price': [22,25000000000000000000000000,27,35]
        }

df2 = DataFrame(Cars,columns= ['Brand', 'Price'])

print (df2)



def outliers_modified_z_score(ys,df,threshold):
    threshold=threshold
    median_y = np.median(ys)
    median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys])
    modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y
                         for y in ys]
    df["Modified-Z-Scores"] = modified_z_scores
    my_list = list()
    for i, row in df.iterrows():
        if row['Modified-Z-Scores'] > threshold or row['Modified-Z-Scores'] < -threshold:
            my_list.append(i)
    return my_list

outlier_datapoints = outliers_modified_z_score(dfVariazioni["Unita'OrganizzativaDirigenziale"], dfVariazioni, 70000)
print(outlier_datapoints)


