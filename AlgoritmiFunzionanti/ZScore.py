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
dataset = pd.read_csv('catastopassicarraichiaiasanferdinandoposillipo.csv',error_bad_lines=False,sep=";",encoding="ISO-8859-1")



from pandas import DataFrame

Cars = {'Brand': ['Honda Civic','Toyota Corolla','Ford Focus','Audi A4','a','b','c','d','e','f','g','h','i','l'],
        'Price': [22,2500,27,35,34,23,78,65,-4600,78,19,28,12,18]
        }

df = DataFrame(Cars,columns= ['Brand', 'Price'])

print (df)

Cars = {'Brand': ['Honda Civic','Toyota Corolla','Ford Focus','Audi A4'],
        'Price': [22,25000000000000000000000000,27,35]
        }

df2 = DataFrame(Cars,columns= ['Brand', 'Price'])

print (df2)

def outliers_z_score(ys,df,threshold):
    threshold = threshold
    mean_y = np.mean(ys)
    stdev_y = np.std(ys)
    z_scores = [(y -  mean_y)/stdev_y for y in ys]
    df["Z-Scores"]=z_scores
    my_list = list()
    for i, row in df.iterrows():
        if row['Z-Scores'] > threshold or row['Z-Scores'] < -threshold:
            print(i, row['Price'])
            my_list.append(i)
    return my_list

outlier_datapoints = outliers_z_score(df["Price"],df,1.5)
#outlier_datapoints2 = outliers_z_score(df2["Price"],df2,1.5)
#outlier_datapoints3=outliers_z_score(dataset["Longitudine"],dataset,3)
print(outlier_datapoints)


'''dataset[""zscore""] = outliers_z_score(dataset[""Unita'OrganizzativaDirigenziale""])
dataset[""is_outlier""] = dataset[""zscore""]
dataset[dataset[""is_outlier""]]

for index,row in dataset.iterrows():
    if(row[""is_outlier""]==True):  
        print(row[""Unita'OrganizzativaDirigenziale""])'''
