from AlgoritmiFunzionanti import lofBuono
from pandas import DataFrame
import pandas as pd
Cars = {'Brand': ['Honda Civic', 'Toyota Corolla', 'Ford Focus', 'Audi A4', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
                  'l'],
        'Price': [22, 2500, 27, 35, 34, 23, 78, 65, -4600, 78, 19, 28, 12, 18]
        }
df = DataFrame(Cars, columns=['Brand', 'Price'])
instances = df.iloc[:, 1].values.reshape(-1, 1)

def test_outliersDBFittizio():
    Y=lofBuono.outliers(10, instances)
    return Y

def test_outlierDBMappe():
    dfMappe = pd.read_csv('catastopassicarraichiaiasanferdinandoposillipo.csv', error_bad_lines=False, sep=';', encoding="ISO-8859-1",nrows=1500)
    instances = dfMappe.iloc[:, 4].values.reshape(-1, 1)
    Y=lofBuono.outliers(10,instances)
    return Y

