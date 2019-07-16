import unittest
from AlgoritmiFunzionanti import LofExample as lf, IQRMethod
from AlgoritmiFunzionanti import ZScore as zs
from pandas import DataFrame
from AlgoritmiFunzionanti import ModifiedZScore
Cars = {
    'Brand': ['Honda Civic', 'Toyota Corolla', 'Ford Focus', 'Audi A4', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
              'l'],
    'Price': [22, 2500, 27, 35, 34, 23, 78, 65, -4600, 78, 19, 28, 12, 18]
}
df = DataFrame(Cars, columns=['Brand', 'Price'])
class TestLOF(unittest.TestCase) :


    def testDBFittizzio(self):

        mylist = list()
        mylist.append(1)
        mylist.append(8)
        self.assertListEqual(zs.outliers_z_score(df["Price"], df, 1.5), mylist)  # ZScore
        self.assertListEqual(lf.DBFittizzio(10, df), mylist)  #LOF
    #self.assertListEqual(IQRMethod.outliers_iqr(df["Price"], df), mylist)  #IQR
        self.assertListEqual(ModifiedZScore.outliers_modified_z_score(df["Price"], df, 4),mylist)





